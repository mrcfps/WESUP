from abc import ABC, abstractmethod

from functools import partial
from skimage.segmentation import slic

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

import config
from utils import empty_tensor
from utils import is_empty_tensor
from .base import BaseModel


class BaseExtractor(ABC):
    """Abstract base class for CNN feature extractors.

    Only `get_conv_layer` method requires to be implemented."""

    def __init__(self, backbone):
        self.backbone = backbone
        self.conv_layers = self.get_conv_layers()
        self.feature_maps = None
        self.fm_size = None
        self.hooks = []

        self._register_hooks()

    def _hook_fn(self, _, input_, output):
        if self.feature_maps is None:
            self.fm_size = (input_[0].size(2), input_[0].size(3))

        output = F.interpolate(output, self.fm_size, mode='bilinear')

        if self.feature_maps is None:
            self.feature_maps = output.squeeze()
        else:
            self.feature_maps = torch.cat(
                (self.feature_maps, output.squeeze()))

    def _register_hooks(self):
        for layer in self.conv_layers:
            self.hooks.append(layer.register_forward_hook(self._hook_fn))

    @abstractmethod
    def get_conv_layers(self):
        """Retrieve all conv layers to extract feature maps from."""

    @property
    def sp_feature_length(self):
        return sum(layer.out_channels for layer in self.conv_layers)

    def extract(self, x):
        """Extract superpixel features."""

        self.feature_maps = None
        _ = self.backbone(x)

        return self.feature_maps


class VGGExtractor(BaseExtractor):
    """VGG network for extracting superpixel features."""

    def get_conv_layers(self):

        return [
            layer
            for layer in self.backbone
            if isinstance(layer, nn.Conv2d)
        ]


class ResNetExtractor(BaseExtractor):
    """ResNet for extracting superpixel features.

    For ResNets, we extract the second conv layer (a.k.a. bottleneck)
    from each bottleneck layer.
    """

    def get_conv_layers(self):
        layers = [self.backbone.conv1]

        layer_no = 1
        while True:
            if not hasattr(self.backbone, f'layer{layer_no}'):
                break
            bottlenecks = getattr(self.backbone, f'layer{layer_no}')
            for bottleneck in bottlenecks:
                layers.append(bottleneck.conv2)

            layer_no += 1

        return layers


class DenseNetExtractor(BaseExtractor):
    """DenseNet for extracting superpixel features.

    For DenseNets, we extract the final conv layer of each denselayer.
    """

    def get_conv_layers(self):
        layers = [self.backbone.conv0]

        block_no = 1
        while True:
            if not hasattr(self.backbone, f'denseblock{block_no}'):
                break
            denseblock = getattr(self.backbone, f'denseblock{block_no}')
            for denselayer in denseblock:
                layers.append(denselayer.conv2)

            block_no += 1

        return layers


class Wessup(BaseModel):
    """WEakly Spervised SUPerpixels."""

    def __init__(self, backbone_name='vgg16', checkpoint=None):
        """Initialize a Wessup model.

        Arguments:
            backbone_name: a string representing the CNN backbone, such as `vgg13` and
                `resnet50` (currently only VGG, ResNet and DenseNet are supported)
            checkpoint: a checkpoint dictionary containing necessary data.
        """

        super().__init__()

        if checkpoint is not None:
            self.backbone_name = checkpoint['backbone']
        else:
            self.backbone_name = backbone_name

        try:
            self.backbone = models.__dict__[backbone_name](pretrained=True)
        except KeyError:
            raise ValueError(f'unsupported backbone {backbone_name}.')

        # remove classifier (if it's VGG or DenseNet)
        if hasattr(self.backbone, 'features'):
            self.backbone = self.backbone.features

        if backbone_name.startswith('vgg'):
            self.extractor = VGGExtractor(self.backbone)
        elif backbone_name.startswith('resnet'):
            self.extractor = ResNetExtractor(self.backbone)
        elif backbone_name.startswith('densenet'):
            self.extractor = DenseNetExtractor(self.backbone)
        else:
            raise ValueError(f'unsupported backbone {backbone_name}.')

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

    @staticmethod
    def label_propagate(X, y_l, threshold=0.95):
        """Perform label propagation with similiarity graph.

        Arguments:
            X: input tensor of size (n_l + n_u, d), where n_l is number of labeled samples,
                n_u is number of unlabeled samples and d is the dimension of input
            y_l: label tensor of size (n_l, c), where c is the number of classes
            threshold: similarity threshold for label propagation

        Returns:
            y_u: propagated label tensor of size (n_u, c)
        """

        # disable gradient computation
        X = X.detach()
        y_l = y_l.detach()

        # number of labeled and unlabeled samples
        n_l = y_l.size(0)
        n_u = X.size(0) - n_l

        # compute similarity matrix W
        Xp = X.view(X.size(0), 1, X.size(1))
        W = torch.exp(-torch.einsum('ijk, ijk->ij', X - Xp, X - Xp))

        # sub-matrix of W containing similarities between labeled and unlabeled samples
        W_ul = W[n_l:, :n_l]

        # max_similarities is the maximum similarity for each unlabeled sample
        # src_indexes is the respective labeled sample index
        max_similarities, src_indexes = W_ul.max(dim=1)

        # initialize y_u with zeros
        y_u = torch.zeros(n_u, y_l.size(1)).to(y_l.device)

        # only propagate labels if maximum similarity is above the threhold
        propagated_samples = max_similarities > threshold
        y_u[propagated_samples] = y_l[src_indexes[propagated_samples]]

        return y_u

    def preprocess(self, *data):
        if self.training:
            img, pixel_mask, point_mask = data
        else:
            img, pixel_mask = data
            point_mask = empty_tensor()

        segments = slic(
            img.squeeze().cpu().numpy().transpose(1, 2, 0),
            n_segments=int(img.size(-2) * img.size(-1) / config.SP_AREA),
            compactness=config.SP_COMPACTNESS,
        )
        segments = torch.LongTensor(segments).to(img.device).unsqueeze(-1)

        pixel_mask = pixel_mask.squeeze()

        if not is_empty_tensor(point_mask):
            point_mask = point_mask.squeeze()
            mask = point_mask
        else:
            mask = pixel_mask

        # ordering of superpixels
        sp_idx_list = range(segments.max() + 1)

        def compute_superpixel_label(mask, segments, sp_idx):
            sp_mask = (mask * (segments == sp_idx).long()).float()
            return sp_mask.sum(dim=(0, 1)) / (sp_mask.sum() + config.EPSILON)

        if not is_empty_tensor(mask):
            # compute labels for each superpixel
            sp_labels = torch.cat([
                compute_superpixel_label(mask, segments, sp_idx).unsqueeze(0)
                for sp_idx in range(segments.max() + 1)
            ])

            # move labeled superpixels to the front of `sp_idx_list`
            labeled_sps = (sp_labels.sum(dim=-1) > 0).nonzero().squeeze()
            unlabeled_sps = (sp_labels.sum(dim=-1) == 0).nonzero().squeeze()
            sp_idx_list = torch.cat([labeled_sps, unlabeled_sps])

            # quantize superpixel labels (e.g., from (0.7, 0.3) to (1.0, 0.0))
            sp_labels = sp_labels[labeled_sps]
            sp_labels = (sp_labels == sp_labels.max(dim=-1, keepdim=True)[0]).float()
        else:  # no supervision provided
            sp_labels = empty_tensor().to(img.device)

        # stacking normalized superpixel segment maps
        sp_maps = torch.cat(
            [(segments == sp_idx).unsqueeze(0) for sp_idx in sp_idx_list])
        sp_maps = sp_maps.squeeze().float()
        sp_maps = sp_maps / sp_maps.sum(dim=(1, 2), keepdim=True)

        return (img, sp_maps), (pixel_mask, sp_labels)

    def forward(self, x):
        """Running a forward pass.

        Args:
            x: a tuple containing input tensor of size (B, C, H, W) and
                stacked superpixel maps with size (N, H, W)

        Returns:
            pred: prediction with size (H, W)
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

        return pred

    def compute_loss(self, pred, target, metrics=None):
        device = pred.device
        _, sp_labels = target

        if self._sp_pred is None:
            raise RuntimeError('You must run a forward pass before computing loss.')

        # total number of superpixels
        total_num = self._sp_pred.size(0)

        # number of labeled superpixels
        labeled_num = sp_labels.size(0)

        def cross_entropy(y_hat, y_true, class_weights=None):
            """Semi-supervised cross entropy loss function.

            Args:
                y_hat: prediction tensor with size (N, C), where C is the number of classes
                y_true: label tensor with size (N, C). A sample won't be counted into loss
                    if its label is all zeros.
                class_weights: class weights tensor with size (C,)

            Returns:
                cross_entropy: cross entropy loss computed only on samples with labels
            """

            # clamp all elements to prevent numerical overflow/underflow
            y_hat = torch.clamp(y_hat, min=config.EPSILON, max=(1 - config.EPSILON))

            # number of samples with labels
            labeled_samples = torch.sum(y_true.sum(dim=1) > 0).float()

            if labeled_samples.item() == 0:
                return torch.tensor(0.).to(device)

            ce = -y_true * torch.log(y_hat)

            if class_weights is not None:
                ce = ce * class_weights.unsqueeze(0)

            return torch.sum(ce) / labeled_samples

        # weighted cross entropy
        ce = partial(cross_entropy,
                     class_weights=torch.Tensor(config.CLASS_WEIGHTS).to(device))

        if labeled_num < total_num:
            # weakly-supervised mode
            propagated_labels = self.label_propagate(self.clf_input_features, sp_labels,
                                                     config.PROPAGATE_THRESHOLD)
            loss = ce(self._sp_pred[:labeled_num], sp_labels)
            propagate_loss = ce(self._sp_pred[labeled_num:], propagated_labels)
            loss += config.PROPAGATE_WEIGHT * propagate_loss
        else:  # fully-supervised mode
            loss = ce(self._sp_pred, sp_labels)

        # clear outdated superpixel prediction
        self._sp_pred = None

        if metrics is not None and isinstance(metrics, dict):
            metrics['labeled_sp_ratio'] = labeled_num / total_num
            metrics['propagated_labels'] = propagated_labels.sum().item()
            metrics['propagate_loss'] = propagate_loss.item()

        return loss

    def _pre_evaluate_hook(self, pred, target):
        pixel_mask, _ = target
        return pred.argmax(dim=-1), pixel_mask.argmax(dim=-1)

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
