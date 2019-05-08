"""
Data loading utilities.
"""

import os
import glob
import random
from functools import partial

import numpy as np
from PIL import Image
from skimage.segmentation import slic

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from torchvision.transforms import ColorJitter

import config


def _list_images(path):
    """Glob all images within a directory."""

    images = []
    for ext in ('jpg', 'jpeg', 'png', 'bmp'):
        images.extend(glob.glob(os.path.join(path, f'*.{ext}')))
    return sorted(images)


class SegmentationDataset(Dataset):
    """One-shot segmentation dataset."""

    def __init__(self, root_dir, rescale_factor=0.5, train=True):
        # path to original images
        self.img_paths = _list_images(os.path.join(root_dir, 'images'))

        # path to mask annotations
        self.mask_paths = None
        if os.path.exists(os.path.join(root_dir, 'masks')):
            self.mask_paths = _list_images(os.path.join(root_dir, 'masks'))

        self.rescale_factor = rescale_factor
        self.train = train

        self.summary()

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx])
        target_height = int(self.rescale_factor * img.height)
        target_width = int(self.rescale_factor * img.width)

        img = img.resize((target_width, target_height), resample=Image.BILINEAR)

        mask = None
        if self.mask_paths is not None:
            mask = Image.open(self.mask_paths[idx])
            mask = mask.resize((target_width, target_height), resample=Image.NEAREST)

        if self.train:
            # perform data augmentation
            img = ColorJitter(.15, .15, .15)(img)
            img, mask = self._transform(img, mask)

        segments = slic(img, n_segments=int(img.width * img.height / config.SP_AREA),
                        compactness=config.SP_COMPACTNESS)
        img = TF.to_tensor(img)
        segments = torch.LongTensor(segments)

        if mask is not None:
            mask = np.array(mask)
            mask = np.concatenate([np.expand_dims(mask == i, -1)
                                   for i in range(config.N_CLASSES)], axis=-1)
            mask = torch.LongTensor(mask.astype('int64'))
            return img, segments, mask

        return img, segments

    def _transform(self, *imgs):
        def _rnd(lower_bound, upper_bound):
            rnd = random.random()
            return lower_bound + (upper_bound - lower_bound) * rnd

        transforms = (
            (1, partial(TF.affine,
                        angle=_rnd(-10, 10),
                        translate=(_rnd(-10, 10), _rnd(-10, 10)),
                        scale=_rnd(1, 1.2),
                        shear=0)
            ),
            (0.5, TF.hflip),
            (0.5, TF.vflip),
        )

        for prob, transform in transforms:
            if random.random() < prob:
                imgs = [transform(img) if img is not None else None for img in imgs]

        if len(imgs) == 1:
            return imgs[0]

        return imgs

    def summary(self):
        mode = 'training' if self.train else 'inference'
        print(f'Segmentation dataset ({mode}) initialized with {len(self.img_paths)} images.')


def get_trainval_dataloaders(root_dir, num_workers):
    """Returns training and validation dataloaders."""

    datasets = {
        'train': SegmentationDataset(os.path.join(root_dir, 'train'),
                                     rescale_factor=config.RESCALE_FACTOR),
        'val': SegmentationDataset(os.path.join(root_dir, 'val'),
                                   rescale_factor=config.RESCALE_FACTOR,
                                   train=False),
    }

    dataloaders = {
        'train': DataLoader(datasets['train'], batch_size=1,
                            shuffle=True, num_workers=num_workers),
        'val': DataLoader(datasets['val'], batch_size=1,
                          shuffle=True, num_workers=num_workers),
    }

    return dataloaders
