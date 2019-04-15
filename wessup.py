import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.transforms import ToTensor

import config


class CNNFeatureExtractor:
    def __init__(self, model, device='cpu'):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.conv_layers = [
            layer
            for layer in self.model
            if isinstance(layer, nn.Conv2d)
        ]
        self.feature_maps = None
        self.fm_size = (config.PATCH_SIZE, config.PATCH_SIZE)
        self.hooks = []

        self._register_hooks()

    def _hook_fn(self, module, input, output):
        output = F.interpolate(output, self.fm_size)

        if self.feature_maps is None:
            self.feature_maps = output.squeeze()
        else:
            self.feature_maps = torch.cat(
                (self.feature_maps, output.squeeze()))

    def _register_hooks(self):
        for layer in self.conv_layers:
            self.hooks.append(layer.register_forward_hook(self._hook_fn))

    @property
    def sp_feature_length(self):
        return sum(layer.out_channels for layer in self.conv_layers)

    def extract(self, x):
        """Extract superpixel features."""

        self.feature_maps = None
        _ = self.model(x)

        return self.feature_maps

    def close(self):
        del self.model
        del self.feature_maps
        for hook in self.hooks:
            hook.remove()


class Wessup(nn.Module):
    def __init__(self, cnn_module, device):
        super().__init__()
        self.cnn_module = cnn_module
        self.extractor = CNNFeatureExtractor(cnn_module, device)

        self.classifier = nn.Sequential(
            nn.Linear(self.extractor.sp_feature_length, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, config.N_CLASSES),
            nn.Softmax(),
        ).to(device)

        self.device = device

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
