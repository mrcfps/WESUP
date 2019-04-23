import csv
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


def _list_images(path):
    """Glob all images within a directory."""

    images = []
    for ext in ('jpg', 'jpeg', 'png', 'bmp'):
        images.extend(glob.glob(os.path.join(path, f'*.{ext}')))
    return images


def _transform_and_crop(img, mask=None):
    """
    Simultaneously apply random transformations to images (and optionally masks).
    """

    if random.random() > 0.5:
        img = TF.hflip(img)
        mask = TF.hflip(mask) if mask else None

    if random.random() > 0.5:
        img = TF.vflip(img)
        mask = TF.vflip(mask) if mask else None

    # possibly some rotations ...

    patch_size = config.PATCH_SIZE
    up = random.randint(0, img.height - patch_size)
    left = random.randint(0, img.width - patch_size)
    img = TF.crop(img, up, left, patch_size, patch_size)
    mask = TF.crop(mask, up, left, patch_size, patch_size) if mask else None

    if mask is None:
        return img, (up, left)

    return (img, mask), (up, left)


def _segment_superpixels(img, label):
    """Segment superpixels of a given image and return segment maps and their labels.

    This function is applicable to two scenarios:

    1. Full annotation: `label` is a mask (`PIL.Image.Image`) with the same height and width as `img`.
        Superpixel maps and their labels will be returned.
    2. Dot annotation: `label` is n_l x 3 array (n_l is the number of points, and each point
        contains information about its x and y coordinates and label). Superpixel maps
        (with n_l labeled superpixels coming first) and n_l labels will be returned.
    """

    segments = slic(img, n_segments=config.SLIC_N_SEGMENTS,
                    compactness=config.SLIC_COMPACTNESS)

    sp_num = segments.max() + 1

    if isinstance(label, Image.Image):
        label = np.array(label)
        label = np.concatenate([np.expand_dims(label == i, -1)
                                for i in range(config.N_CLASSES)], axis=-1)
        sp_labels = np.array([
            (label * np.expand_dims(segments == i, -1)
             ).sum(axis=(0, 1)) / np.sum(segments == i)
            for i in range(sp_num)
        ])
        sp_labels = np.argmax(
            sp_labels == sp_labels.max(axis=-1, keepdims=True), axis=-1)
        sp_idx_list = range(sp_num)
    else:
        labeled_sps, sp_labels = [], []

        for point in label:
            i, j, label = point
            if segments[i, j] not in labeled_sps:
                labeled_sps.append(segments[i, j])

        unlabeled_sps = list(set(np.unique(segments) - set(labeled_sps)))
        sp_idx_list = labeled_sps + unlabeled_sps

    # stacking normalized superpixel segment maps
    sp_maps = np.concatenate([np.expand_dims(segments == idx, 0) for idx in sp_idx_list])
    sp_maps = sp_maps / sp_maps.sum(axis=0, keepdims=True)

    return sp_maps, sp_labels


class FullAnnotationDataset(Dataset):
    """Segmentation dataset with mask-level (full) annotation."""

    def __init__(self, root_dir, train=True):
        self.img_paths = _list_images(os.path.join(root_dir, 'images'))
        self.mask_paths = _list_images(os.path.join(root_dir, 'masks'))
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
            (img, mask), _ = _transform_and_crop(img, mask)

        sp_maps, sp_labels = _segment_superpixels(img, mask)

        # convert to tensors
        img = TF.to_tensor(img)
        mask = torch.LongTensor(np.array(mask))
        sp_maps = torch.Tensor(sp_maps)
        sp_labels = torch.LongTensor(sp_labels)

        return img, mask, sp_maps, sp_labels


class DotAnnotationDataset(Dataset):
    """Segmentation dataset with dot annotation."""

    def __init__(self, root_dir):
        self.img_paths = _list_images(os.path.join(root_dir, 'images'))
        self.label_paths = glob.glob(os.path.join(root_dir, 'labels', '*.csv'))

        patch_area = config.PATCH_SIZE ** 2
        img = Image.open(self.img_paths[0])
        img_area = img.height * img.width
        self.patches_per_img = int(np.round(img_area / patch_area))

    def __len__(self):
        return len(self.img_paths) * self.patches_per_img

    def __getitem__(self, idx):
        idx = idx // self.patches_per_img
        img, (up, left) = _transform_and_crop(Image.open(self.img_paths[idx]))

        with open(self.label_paths[idx]) as fp:
            reader = csv.reader(fp)
            label = np.array([[int(d) for d in point] for point in reader])

        # subtract offsets from top and left
        label[:, 0] -= up
        label[:, 1] -= left

        sp_maps, sp_labels = _segment_superpixels(img, label)

        # convert to tensors
        img = TF.to_tensor(img)
        sp_maps = torch.Tensor(sp_maps)
        sp_labels = torch.LongTensor(sp_labels)

        # the second return value is the missing mask for convenience
        return img, None, sp_maps, sp_labels


def get_trainval_dataloaders(root_dir, num_workers):
    """Returns training and validation dataloaders."""

    train_dir = os.path.join(root_dir, 'train')
    val_dir = os.path.join(root_dir, 'val')

    # if `labels` directory is present in `root_dir`, then it's dot annotation mode
    is_dot_anno = os.path.exists(os.path.join(train_dir, 'labels'))

    datasets = {
        'train': DotAnnotationDataset(train_dir) if is_dot_anno else FullAnnotationDataset(train_dir),
        'val': FullAnnotationDataset(val_dir, train=False),
    }

    dataloaders = {
        'train': DataLoader(datasets['train'], batch_size=1,
                            shuffle=True, num_workers=num_workers),
        'val': DataLoader(datasets['val'], batch_size=1,
                          shuffle=True, num_workers=num_workers),
    }

    return dataloaders
