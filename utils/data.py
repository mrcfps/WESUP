"""
Data loading utilities.
"""

import csv
import os
import glob
from functools import partial

import numpy as np
from skimage.io import imread
from skimage.segmentation import slic

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF

import config
from .tile import pad_image
from .tile import compute_patches_grid_shape


def _list_images(path):
    """Glob all images within a directory."""

    images = []
    for ext in ('jpg', 'jpeg', 'png', 'bmp'):
        images.extend(glob.glob(os.path.join(path, f'*.{ext}')))
    return sorted(images)


def _transform_and_crop(*arrs):
    """
    Simultaneously apply random transformations to multiple images.
    """

    def crop(arr, rand1, rand2):
        patch_size = config.PATCH_SIZE
        up = int(rand1 * (arr.shape[0] - patch_size))
        left = int(rand2 * (arr.shape[1] - patch_size))
        return arr[up:up + patch_size, left:left + patch_size]

    crop_partial = partial(crop, rand1=np.random.random(), rand2=np.random.random())

    # transforms to apply (the final `copy` is to avert negative-strided arrays)
    transforms = [(0.5, np.fliplr), (0.5, np.flipud), (1, crop_partial), (1, np.copy)]

    for prob, transform in transforms:
        if np.random.random() < prob:
            arrs = [transform(arr) if arr is not None else None for arr in arrs]

    if len(arrs) == 0:
        return arrs[0]

    return arrs


class PatchDataset(Dataset):
    """Dataset with cropped patches used for training and validation.

    Return values:
        img: image patch tensor with size (3, H, W)
        segments: superpixel segmentation map with size (H, W)
        mask: pixel-level annotation with size (H, W, n_classes) if given, else
            a zero tensor will be returned
        point_mask: point-level annotation with size (H, W, n_classes) if given,
            else a zero tensor will be returned
    """

    def __init__(self, root_dir, train=True):
        # path to images
        self.img_paths = _list_images(os.path.join(root_dir, 'images'))

        # path to mask annotations
        self.mask_paths = None

        # path to dot annotations
        self.label_paths = None

        if os.path.exists(os.path.join(root_dir, 'masks')):
            self.mask_paths = _list_images(os.path.join(root_dir, 'masks'))

        if os.path.exists(os.path.join(root_dir, 'labels')):
            self.label_paths = sorted(glob.glob(os.path.join(root_dir, 'labels', '*.csv')))

        self.train = train

        # compute how many patches are sampled for each image
        patch_area = config.PATCH_SIZE ** 2
        img = imread(self.img_paths[0])
        img_area = img.shape[0] * img.shape[1]
        self.patches_per_img = int(np.round(img_area / patch_area))

        self.summary()

    def __len__(self):
        return len(self.img_paths) * self.patches_per_img

    def __getitem__(self, patch_idx):
        img_idx = patch_idx // self.patches_per_img
        img = imread(self.img_paths[img_idx])
        mask, point_mask = None, None

        if self.mask_paths is not None:
            mask = imread(self.mask_paths[img_idx])
            mask = np.concatenate([np.expand_dims(mask == i, -1)
                                   for i in range(config.N_CLASSES)], axis=-1)
            mask = mask.astype('int64')

        if self.label_paths is not None:
            with open(self.label_paths[img_idx]) as fp:
                reader = csv.reader(fp)
                points = np.array([[int(d) for d in point] for point in reader])

            # compute point mask
            point_mask = np.zeros((img.shape[0], img.shape[1], config.N_CLASSES))
            for i, j, class_ in points:
                c = np.zeros(config.N_CLASSES)
                c[class_] = 1
                point_mask[i, j] = c

        if self.train:
            img, mask, point_mask = _transform_and_crop(img, mask, point_mask)

        segments = slic(img, n_segments=int(img.shape[0] * img.shape[1] / config.SP_AREA),
                        compactness=config.SP_COMPACTNESS)

        # convert to tensors
        img = TF.to_tensor(img)
        segments = torch.LongTensor(segments)

        if mask is not None:
            mask = torch.LongTensor(mask)

        if point_mask is not None:
            point_mask = torch.LongTensor(point_mask)

        data = (img, segments, mask, point_mask)

        return tuple(datum if datum is not None else torch.tensor(0) for datum in data)

    def summary(self):
        print(f'{"Training" if self.train else "Validation"} set initialized with {len(self.img_paths)} images ({len(self)} patches).')

        if self.mask_paths or self.label_paths:
            print(f'Supervision mode: {"point" if self.label_paths is not None else "mask"}')


class WholeImageDataset(Dataset):
    """Dataset with whole images for inference."""

    def __init__(self, root_dir):
        self.img_paths = _list_images(os.path.join(root_dir, 'images'))

        if os.path.exists(os.path.join(root_dir, 'masks')):
            self.masks = [
                np.array(imread(mask_path))
                for mask_path in _list_images(os.path.join(root_dir, 'masks'))
            ]
        else:
            self.masks = None

        # patches grid shape for each image
        self.patches_grids = [
            compute_patches_grid_shape(imread(img_path),
                                       config.PATCH_SIZE, config.INFER_STRIDE)
            for img_path in self.img_paths
        ]

        # number of patches for each image
        self.patches_nums = [n_h * n_w for n_h, n_w in self.patches_grids]

        # sequence for identifying image index from patch index
        self.patches_numseq = np.cumsum(self.patches_nums)

        self.summary()

    def __len__(self):
        return sum(self.patches_nums)

    def __getitem__(self, patch_idx):
        img_idx = self.patch2img(patch_idx)
        img = np.array(imread(self.img_paths[img_idx]))
        img = pad_image(img, config.PATCH_SIZE, config.INFER_STRIDE)

        if img_idx > 0:
            # patch index WITHIN this image
            patch_idx -= self.patches_numseq[img_idx - 1]

        _, n_w = self.patches_grids[img_idx]
        up = (patch_idx // n_w) * config.INFER_STRIDE
        left = (patch_idx % n_w) * config.INFER_STRIDE

        patch = img[up:up + config.PATCH_SIZE, left:left + config.PATCH_SIZE]
        segments = slic(patch, n_segments=int(patch.shape[0] * patch.shape[1] / config.SP_AREA),
                        compactness=config.SP_COMPACTNESS)

        return TF.to_tensor(patch), torch.LongTensor(segments)

    def summary(self):
        print(f'Whole image dataset initialized with {len(self.img_paths)} images ({len(self)} patches).')

    def patch2img(self, patch_idx):
        """Identify which image this patch belongs to."""

        return np.argmax(self.patches_numseq > patch_idx)


def get_trainval_dataloaders(root_dir, num_workers):
    """Returns training and validation dataloaders."""

    datasets = {
        'train': PatchDataset(os.path.join(root_dir, 'train')),
        'val': PatchDataset(os.path.join(root_dir, 'val'), train=False),
    }

    dataloaders = {
        'train': DataLoader(datasets['train'], batch_size=1,
                            shuffle=True, num_workers=num_workers),
        'val': DataLoader(datasets['val'], batch_size=1,
                          shuffle=True, num_workers=num_workers),
    }

    return dataloaders
