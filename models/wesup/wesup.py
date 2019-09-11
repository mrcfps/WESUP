import os.path as osp
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from skimage.segmentation import slic

from utils import empty_tensor
from utils import is_empty_tensor
from utils.data import SegmentationDataset
from utils.data import PointSupervisionDataset
from ..base import BaseModel
from .common import cross_entropy
from .common import preprocess_superpixels
from .common import label_propagate
from .config import config


class WESUP(BaseModel):
    """WEakly Spervised SUPerpixels."""

    def __init__(self, checkpoint=None):
        """Initialize a WESUP model.

        Arguments:
            checkpoint: a checkpoint dictionary containing necessary data.
        """

        super().__init__()

        self.config = config
        self.backbone = models.vgg16(pretrained=True).features

        if self.config.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # sum of channels of all feature maps
        self.fm_channels_sum = 0

        # side convolution layers after each conv feature map
        for layer in self.backbone:
            if isinstance(layer, nn.Conv2d):
                layer.register_forward_hook(self._hook_fn)
                setattr(self, f'side_conv{self.fm_channels_sum}',
                        nn.Conv2d(layer.out_channels, layer.out_channels // 2, 1))
                self.fm_channels_sum += layer.out_channels // 2

        # fully-connected layers for dimensionality reduction
        self.fc_layers = nn.Sequential(
            nn.Linear(self.fm_channels_sum, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 32),
            nn.ReLU()
        )

        # final softmax classifier
        self.classifier = nn.Sequential(
            nn.Linear(32, 2),
            nn.Softmax(dim=1)
        )

        # store conv feature maps
        self.feature_maps = None

        # spatial size of first feature map
        self.fm_size = None

        # label propagation input features
        self._sp_features = None

        # superpixel predictions (internally tracked to compute loss)
        self._sp_pred = None

        if checkpoint is not None:
            self.load_state_dict(checkpoint['model_state_dict'])

    def _hook_fn(self, _, input_, output):
        if self.feature_maps is None:
            self.fm_size = (input_[0].size(2), input_[0].size(3))
            side_conv_name = 'side_conv0'
        else:
            side_conv_name = f'side_conv{len(self.feature_maps)}'

        output = getattr(self, side_conv_name)(output.clone())
        output = F.interpolate(output, self.fm_size, mode='bilinear', align_corners=True)

        if self.feature_maps is None:
            self.feature_maps = output.squeeze()
        else:
            self.feature_maps = torch.cat(
                (self.feature_maps, output.squeeze()))

    def get_default_dataset(self, root_dir, train=True, proportion=1.0):
        slic_params = {
            'area': self.config.sp_area,
            'compactness': self.config.sp_compactness,
        }
        if train:
            if osp.exists(osp.join(root_dir, 'points')):
                return PointSupervisionDataset(root_dir, proportion=proportion,
                                               multiscale_range=self.config.multiscale_range)
            return SegmentationDataset(root_dir, proportion=proportion,
                                       multiscale_range=self.config.multiscale_range)
        return SegmentationDataset(root_dir, rescale_factor=self.config.rescale_factor, train=False)

    def get_default_optimizer(self, checkpoint=None):
        optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=1e-3,
            momentum=self.config.momentum,
            weight_decay=self.config.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', patience=20, factor=0.5, min_lr=1e-5, verbose=True)

        if checkpoint is not None:
            try:
                # load previous optimizer states
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            except KeyError:  # if not present in checkpoint, ignore it
                pass

        return optimizer, scheduler

    def preprocess(self, *data, device='cpu'):
        data = [datum.to(device) for datum in data]
        if len(data) == 3:
            img, pixel_mask, point_mask = data
        else:
            img, pixel_mask = data
            point_mask = empty_tensor()

        segments = slic(
            img.squeeze().cpu().numpy().transpose(1, 2, 0),
            n_segments=int(img.size(-2) * img.size(-1) / self.config.sp_area),
            compactness=self.config.sp_compactness,
        )
        segments = torch.LongTensor(segments).to(img.device)

        if point_mask is not None and not is_empty_tensor(point_mask):
            mask = point_mask.squeeze()
        elif pixel_mask is not None and not is_empty_tensor(pixel_mask):
            mask = pixel_mask.squeeze()
        else:
            mask = None

        sp_maps, sp_labels = preprocess_superpixels(segments, mask)

        return (img, sp_maps), (pixel_mask, sp_labels)

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
        _ = self.backbone(x)
        x = self.feature_maps
        self.feature_maps = None
        x = x.view(x.size(0), -1)

        # calculate features for each superpixel
        sp_maps = sp_maps.view(sp_maps.size(0), -1)
        x = torch.mm(sp_maps, x.t())

        # reduce superpixel feature dimensions with fully connected layers
        x = self.fc_layers(x)
        self._sp_features = x

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
                     class_weights=torch.Tensor(self.config.class_weights).to(device))

        if labeled_num < total_num:
            # weakly-supervised mode
            loss = ce(self._sp_pred[:labeled_num], sp_labels)

            if self.config.enable_propagation:
                propagated_labels = label_propagate(self._sp_features, sp_labels,
                                                    threshold=self.config.propagate_threshold)

                propagate_loss = ce(self._sp_pred[labeled_num:], propagated_labels)
                loss += self.config.propagate_weight * propagate_loss

            if metrics is not None and isinstance(metrics, dict):
                metrics['labeled_sp_ratio'] = labeled_num / total_num
                if self.config.enable_propagation:
                    metrics['propagated_labels'] = propagated_labels.sum().item()
                    metrics['propagate_loss'] = propagate_loss.item()
        else:  # fully-supervised mode
            loss = ce(self._sp_pred, sp_labels)

        # clear outdated superpixel prediction
        self._sp_pred = None

        return loss

    def postprocess(self, pred, target=None):
        pred = pred.round().long()
        if target is not None:
            pixel_mask, _ = target
            return pred, pixel_mask.argmax(dim=1)
        return pred

    def save_checkpoint(self, ckpt_path, **kwargs):
        """Save model checkpoint."""

        torch.save({
            'model_state_dict': self.state_dict(),
            **kwargs,
        }, ckpt_path)
