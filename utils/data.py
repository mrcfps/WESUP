import os
import random
import glob

import numpy as np
from PIL import Image
from skimage.segmentation import slic

import torch
from torch.utils.data import Dataset, DataLoader
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


class SuperpixelDataset(Dataset):
    """Superpixel generation and labeling dataset."""

    def __init__(self, root_dir, train=True):
        data_dir = os.path.join(root_dir, 'train' if train else 'val')
        self.img_paths = glob.glob(os.path.join(data_dir, 'images', '*.png'))
        self.mask_paths = glob.glob(os.path.join(data_dir, 'masks', '*.png'))
        self.train = train

        patch_area = config.PATCH_SIZE ** 2
        img = Image.open(self.img_paths[0])
        img_area = img.height * img.width
        self.patches_per_img = int(np.round(img_area / patch_area))

    def __len__(self):
        return len(self.img_paths) * self.patches_per_img

    def __getitem__(self, idx):
        idx = idx // self.patches_per_img
        img = Image.open(self.img_paths[idx])
        mask = Image.open(self.mask_paths[idx])

        if self.train:
            img, mask = transform_img_and_mask(img, mask)

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

        # convert to tensors
        img = TF.to_tensor(img)
        mask = TF.to_tensor(mask.astype('int')).long()
        sp_maps = torch.Tensor(sp_maps)
        sp_labels = torch.LongTensor(sp_labels)

        return img, mask, sp_maps, sp_labels


def get_trainval_dataloaders(root_dir):
    """Returns training and validation dataloaders."""

    datasets = {
        'train': SuperpixelDataset(root_dir, train=True),
        'val': SuperpixelDataset(root_dir, train=False),
    }
    dataloaders = {
        'train': DataLoader(datasets['train'], batch_size=1,
                            shuffle=True, num_workers=os.cpu_count() // 2),
        'val': DataLoader(datasets['val'], batch_size=1,
                          shuffle=True, num_workers=os.cpu_count() // 2),
    }

    return dataloaders
