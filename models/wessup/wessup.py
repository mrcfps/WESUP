from functools import partial
from skimage.segmentation import slic

import torch
import torch.nn as nn
from torchvision import models

from utils import empty_tensor
from utils import is_empty_tensor
from utils.data import SegmentationDataset
from utils.data import PointSupervisionDataset
from ..base import BaseModel
from .common import cross_entropy
from .common import preprocess_superpixels
from .common import label_propagate
from .extractors import VGGExtractor
from .extractors import ResNetExtractor
from .extractors import DenseNetExtractor
from .config import config


class Wessup(BaseModel):
    """WEakly Spervised SUPerpixels."""

    def __init__(self, checkpoint=None):
        """Initialize a Wessup model.

        Arguments:
            checkpoint: a checkpoint dictionary containing necessary data.
        """

        super().__init__()

        if checkpoint is not None:
            self.backbone_name = checkpoint['backbone']
        else:
            self.backbone_name = config.backbone

        try:
            self.backbone = models.__dict__[self.backbone_name](pretrained=True)
        except KeyError:
            raise ValueError(f'unsupported backbone {self.backbone_name}.')

        # remove classifier (if it's VGG or DenseNet)
        if hasattr(self.backbone, 'features'):
            self.backbone = self.backbone.features

        if self.backbone_name.startswith('vgg'):
            self.extractor = VGGExtractor(self.backbone)
        elif self.backbone_name.startswith('resnet'):
            self.extractor = ResNetExtractor(self.backbone)
        elif self.backbone_name.startswith('densenet'):
            self.extractor = DenseNetExtractor(self.backbone)
        else:
            raise ValueError(f'unsupported backbone {self.backbone_name}.')

        # fully-connected layers for dimensionality reduction
        self.fc_layers = nn.Sequential(
            nn.Linear(self.extractor.sp_feature_length, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 32),
            nn.ReLU()
        )

        # final softmax classifier
        self.classifier = nn.Sequential(
            nn.Linear(32, 2),
            nn.Softmax()
        )

        # label propagation input features
        self.clf_input_features = None
        self.fc_layers.register_forward_hook(self._hook_fn)

        # superpixel predictions (internally tracked to compute loss)
        self._sp_pred = None

        if checkpoint is not None:
            self.load_state_dict(checkpoint['model_state_dict'])

        self.summary()

    def _hook_fn(self, module, input, output):
        self.clf_input_features = output

    def get_default_dataset(self, root_dir, train=True):
        if train:
            return PointSupervisionDataset(root_dir, rescale_factor=config.rescale_factor)

        return SegmentationDataset(root_dir, rescale_factor=config.rescale_factor, train=False)

    def get_default_optimizer(self, checkpoint=None):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=1e-3,
            momentum=config.momentum,
            weight_decay=config.weight_decay
        )

        if checkpoint is not None:
            # load previous optimizer states
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'max', factor=0.5, min_lr=5e-5, verbose=True)

        return optimizer, scheduler

    def preprocess(self, *data):
        if self.training:
            img, pixel_mask, point_mask = data
        else:
            img, pixel_mask = data
            point_mask = empty_tensor()

        segments = slic(
            img.squeeze().cpu().numpy().transpose(1, 2, 0),
            n_segments=int(img.size(-2) * img.size(-1) / config.sp_area),
            compactness=config.sp_compactness,
        )
        segments = torch.LongTensor(segments).to(img.device)

        pixel_mask = pixel_mask.squeeze()

        if not is_empty_tensor(point_mask):
            point_mask = point_mask.squeeze()
            mask = point_mask
        else:
            mask = pixel_mask

        sp_maps, sp_labels = preprocess_superpixels(segments, mask)

        return (img, sp_maps), (pixel_mask.unsqueeze(0), sp_labels)

    def forward(self, x):
        """Running a forward pass.

        Args:
            x: a tuple containing input tensor of size (1, C, H, W) and
                stacked superpixel maps with size (N, H, W)

        Returns:
            pred: prediction with size (1, H, W)
        """

        x, sp_maps = x
        n_superpixels, height, width = sp_maps.size()

        # extract conv feature maps and flatten
        x = self.extractor.extract(x)
        x = x.view(x.size(0), -1)

        # calculate features for each superpixel
        sp_maps = sp_maps.view(sp_maps.size(0), -1)
        x = torch.mm(sp_maps, x.t())

        # reduce superpixel feature dimensions with fully connected layers
        x = self.fc_layers(x)

        # classify each superpixel
        self._sp_pred = self.classifier(x)

        # flatten sp_maps to one channel
        sp_maps = sp_maps.view(n_superpixels, height, width).argmax(dim=0)

        # initialize prediction mask
        pred = torch.zeros(height, width, self._sp_pred.size(1))
        pred = pred.to(sp_maps.device)

        for sp_idx in range(sp_maps.max().item() + 1):
            pred[sp_maps == sp_idx] = self._sp_pred[sp_idx]

        return pred.unsqueeze(0)[..., 1]

    def compute_loss(self, pred, target, metrics=None):
        device = pred.device
        _, sp_labels = target

        if self._sp_pred is None:
            raise RuntimeError('You must run a forward pass before computing loss.')

        # total number of superpixels
        total_num = self._sp_pred.size(0)

        # number of labeled superpixels
        labeled_num = sp_labels.size(0)

        # weighted cross entropy
        ce = partial(cross_entropy,
                     class_weights=torch.Tensor(config.class_weights).to(device))

        if labeled_num < total_num:
            # weakly-supervised mode
            propagated_labels = label_propagate(self.clf_input_features, sp_labels,
                                                config.propagate_threshold)
            loss = ce(self._sp_pred[:labeled_num], sp_labels)
            propagate_loss = ce(self._sp_pred[labeled_num:], propagated_labels)
            loss += config.propagate_weight * propagate_loss
        else:  # fully-supervised mode
            loss = ce(self._sp_pred, sp_labels)

        # clear outdated superpixel prediction
        self._sp_pred = None

        if metrics is not None and isinstance(metrics, dict):
            metrics['labeled_sp_ratio'] = labeled_num / total_num
            metrics['propagated_labels'] = propagated_labels.sum().item()
            metrics['propagate_loss'] = propagate_loss.item()

        return loss

    def postprocess(self, pred, target):
        pixel_mask, _ = target
        return pred.round().long(), pixel_mask.argmax(dim=1)

    def save_checkpoint(self, ckpt_path, **kwargs):
        """Save model checkpoint."""

        torch.save({
            'backbone': self.backbone_name,
            'model_state_dict': self.state_dict(),
            **kwargs,
        }, ckpt_path)
        print(f'Checkpoint saved to {ckpt_path}.')

    def summary(self):
        """Print summary information."""

        print(
            f'Wessup initialized with {self.backbone_name} backbone '
            f'({len(self.extractor.conv_layers)} conv layers).')
        print(f'Superpixel feature length: {self.extractor.sp_feature_length}')
