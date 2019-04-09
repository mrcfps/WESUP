import numpy as np

import torch
import torch.nn.functional as F
from torchvision.models import vgg16
from torchvision.transforms import ToTensor

from skimage.segmentation import slic

import config


class VGG16FeatureExtractor:
    def __init__(self):
        self.device = torch.device(config.DEVICE)
        self.model = vgg16(pretrained=True).to(self.device)
        self.conv_layers = [
            layer
            for layer in self.model.features
            if isinstance(layer, torch.nn.Conv2d)
        ]
        self.feature_maps = None
        self.fm_size = None
        self.hooks = []

        # Turn off grad computing.
        self.model.eval()

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

    def extract(self, img, mask=None):
        """Extract superpixel features.

        Parameters
        ==========
        img: input image of `PIL.Image`
        mask (optional): mask image of `PIL.Image`

        Returns
        =======
        sp_features: superpixel features tensor (n_superpixels x n_features)
        sp_labels: superpixel labels tensor (n_superpixels)
        """

        self.feature_maps = None
        self.fm_size = (img.height, img.width)
        input_tensor = ToTensor()(img).unsqueeze(0).to(self.device)
        _ = self.model.features(input_tensor)

        # extract conv feature maps
        self.feature_maps = self.feature_maps.view(
            self.feature_maps.size(0), -1)

        # SLIC superpixel segmentation
        segments = slic(img, n_segments=config.SLIC_N_SEGMENTS,
                        compactness=config.SLIC_COMPACTNESS)
        sp_num = segments.max() + 1

        # stacking normalized superpixel segment maps
        sp_maps = np.concatenate([np.expand_dims(segments == i, 0)
                                  for i in range(sp_num)])
        sp_maps = torch.Tensor(sp_maps.astype('float32')).to(self.device)
        sp_maps = sp_maps.view(-1, sp_num)
        sp_maps = sp_maps / sp_maps.sum(dim=0, keepdim=True)

        sp_features = torch.mm(self.feature_maps, sp_maps).t()

        if mask is not None:
            mask = np.array(mask) > 0

            # compute labels for each superpixel
            sp_labels = torch.Tensor([mask[segments == i].mean().round()
                                      for i in range(sp_num)])
            return sp_features, sp_labels
        else:
            return sp_features

    def close(self):
        del self.model
        del self.feature_maps
        for hook in self.hooks:
            hook.remove()
