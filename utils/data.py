import os
import random
import glob

import numpy as np
from PIL import Image
from skimage.segmentation import slic

import torch
import torchvision.transforms.functional as TF

import config


def transform_img_and_mask(img, mask):
    if random.random() > 0.5:
        img = TF.hflip(img)
        mask = TF.hflip(mask)

    if random.random() > 0.5:
        img = TF.vflip(img)
        mask = TF.vflip(mask)

    # possibly some rotations ...

    patch_size = config.PATCH_SIZE
    up = random.randint(0, img.height - patch_size)
    left = random.randint(0, img.width - patch_size)
    img = TF.crop(img, up, left, patch_size, patch_size)
    mask = TF.crop(mask, up, left, patch_size, patch_size)

    return img, mask


class SuperpixelDataset(torch.utils.data.Dataset):
    """Superpixel generation and labeling dataset."""

    def __init__(self, root_dir, transform=None):
        self.img_paths = glob.glob(os.path.join(root_dir, 'images', '*.png'))
        self.mask_paths = glob.glob(os.path.join(root_dir, 'masks', '*.png'))
        self.transform = transform or transform_img_and_mask

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx])
        mask = Image.open(self.mask_paths[idx])

        if self.transform is not None:
            img, mask = self.transform(img, mask)

        segments = slic(img, n_segments=config.SLIC_N_SEGMENTS,
                        compactness=config.SLIC_COMPACTNESS)

        sp_num = segments.max() + 1

        # stacking normalized superpixel segment maps
        sp_maps = np.concatenate([np.expand_dims(segments == i, 0)
                                  for i in range(sp_num)])
        sp_maps = sp_maps / sp_maps.sum(axis=0, keepdims=True)

        mask = np.array(mask)
        mask = np.concatenate([np.expand_dims(segments == i, -1)
                               for i in range(config.N_CLASSES)], axis=-1)
        sp_labels = np.array([
            (mask * np.expand_dims(segments == i, -1)
             ).sum(axis=(0, 1)) / np.sum(segments == i)
            for i in range(sp_num)
        ])
        sp_labels = np.argmax(
            sp_labels == sp_labels.max(axis=-1, keepdims=True), axis=-1)

        return TF.to_tensor(img), torch.Tensor(sp_maps), torch.LongTensor(sp_labels)
