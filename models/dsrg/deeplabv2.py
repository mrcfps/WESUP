import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16


class ASPP(nn.Module):
    """
    Atrous spatial pyramid pooling (ASPP) module.
    """

    def __init__(self, in_channels, out_channels, rates=(6, 12, 18, 24)):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.rates = rates

        for idx, rate in enumerate(rates):
            setattr(self, f'branch_{idx}', self._build_branch(rates[idx]))

    def _build_branch(self, rate):
        branch = nn.Sequential(
            nn.Conv2d(self.in_channels, 1024, 3, stride=1,
                      padding=rate, dilation=rate),
            nn.ReLU(inplace=False),
            nn.Dropout2d(p=0.5),
            nn.Conv2d(1024, 1024, 1, stride=1),
            nn.ReLU(inplace=False),
            nn.Dropout2d(p=0.5),
            nn.Conv2d(1024, self.out_channels, 1, stride=1),
        )

        # special initialization for the last conv layer
        nn.init.normal_(branch[-1].weight, mean=0, std=0.01)
        nn.init.constant_(branch[-1].bias, 0)

        return branch

    def forward(self, x):
        return sum(getattr(self, f'branch_{idx}')(x) for idx in range(len(self.rates)))


class DeepLabV2(nn.Module):
    """
    DeepLab v2: Dilated VGG-16 + ASPP (output downscaled by 8x).
    """

    def __init__(self, n_classes):
        super().__init__()
        self.vgg_features = self._build_tweaked_vgg()
        self.aspp = ASPP(512, n_classes, rates=(6, 12, 18, 24))

    def _build_tweaked_vgg(self):
        model = vgg16(pretrained=True).features

        for layer in model:
            if isinstance(layer, nn.MaxPool2d):
                layer.kernel_size = 3
                layer.stride = 2
                layer.padding = 1

        # set stride to 1 for the last two max-pooling layers
        model[23].stride = 1
        model[30].stride = 1

        # tweak dilations and paddings of conv5_* layers
        for layer_idx in (24, 26, 28):
            model[layer_idx].padding = (2, 2)
            model[layer_idx].dilation = (2, 2)

        return model

    def forward(self, x):
        x = F.avg_pool2d(self.vgg_features(x), 3, stride=1, padding=1)
        return F.softmax(self.aspp(x), dim=1)
