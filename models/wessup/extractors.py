from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F


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
