from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

import config


class BaseExtractor(ABC):
    """Abstract base class for CNN feature extractors.

    Only `get_conv_layer` method requires to be implemented."""

    def __init__(self, backbone):
        self.backbone = backbone
        self.conv_layers = self.get_conv_layers()

        print(f'Wessup extractor with {len(self.conv_layers)} conv layers.')
        print(f'Resulting in superpixel features of length {self.sp_feature_length}.')

        self.feature_maps = None
        self.fm_size = (config.PATCH_SIZE, config.PATCH_SIZE)
        self.hooks = []

        self._register_hooks()

    def _hook_fn(self, module, input, output):
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
        pass

    @property
    def sp_feature_length(self):
        return sum(layer.out_channels for layer in self.conv_layers)

    def extract(self, x):
        """Extract superpixel features."""

        self.feature_maps = None
        _ = self.backbone(x)

        return self.feature_maps

    def close(self):
        for hook in self.hooks:
            hook.remove()


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


class Wessup(nn.Module):
    def __init__(self, backbone_name):
        """Initialize a Wessup model.

        Arguments:
            backbone_name: a string representing the CNN backbone, such as `vgg13` and
                `resnet50` (currently only VGG, ResNet and DenseNet are supported)
        """

        super().__init__()
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

        self.classifier = nn.Sequential(
            self._build_fc_layer(self.extractor.sp_feature_length, 1024),
            self._build_fc_layer(1024, 1024),
            self._build_fc_layer(1024, 32),
            nn.Linear(32, config.N_CLASSES),
            nn.Softmax()
        )

        # label propagation input features
        self.lp_input_features = None
        self.classifier[-2].register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, input, output):
        self.lp_input_features = input[0]

    def forward(self, x, sp_maps):
        # extract conv feature maps and flatten
        x = self.extractor.extract(x)
        x = x.view(x.size(0), -1)

        # calculate features for each superpixel
        sp_maps = sp_maps.view(sp_maps.size(0), -1)
        x = torch.mm(sp_maps, x.t())

        # classify each superpixel with an MLP
        x = self.classifier(x)

        return x

    def _build_fc_layer(self, in_features, out_features):
        return nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.ReLU(),
            nn.BatchNorm1d(out_features),
            nn.Dropout()
        )
