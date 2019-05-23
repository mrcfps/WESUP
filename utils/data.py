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
from skimage.feature import greycomatrix
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

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


def _transform_multiple_images(*imgs):
    """Apply identical transformations to multiple PIL images."""

    def rnd(lower_bound, upper_bound):
        rnd = random.random()
        return lower_bound + (upper_bound - lower_bound) * rnd

    def elastic_transform(image, alpha, sigma, spline_order=1, mode='nearest', random_state=42):
        """Elastic deformation of image as described in [Simard2003]_.
        .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
        Convolutional Neural Networks applied to Visual Document Analysis", in
        Proc. of the International Conference on Document Analysis and
        Recognition, 2003.
        """

        np.random.seed(random_state)
        image = np.array(image)
        shape = image.shape[:2]

        dx = gaussian_filter((np.random.rand(*shape) * 2 - 1),
                             sigma, mode="constant", cval=0) * alpha
        dy = gaussian_filter((np.random.rand(*shape) * 2 - 1),
                             sigma, mode="constant", cval=0) * alpha

        x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
        indices = [np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1))]

        if image.ndim == 2:
            result = map_coordinates(image, indices, order=spline_order, mode=mode).reshape(shape)
        else:
            result = np.empty_like(image)
            for i in range(image.shape[2]):
                result[:, :, i] = map_coordinates(
                    image[:, :, i], indices, order=spline_order, mode=mode).reshape(shape)

        return Image.fromarray(result)

    transforms = (
        (1, partial(elastic_transform,
                    alpha=rnd(0, 1000),
                    sigma=rnd(20, 50),
                    random_state=np.random.randint(100))
        ),
        (1, partial(TF.affine,
                    angle=rnd(-20, 20),
                    translate=(rnd(-20, 20), rnd(-20, 20)),
                    scale=rnd(1, 1.4),
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


def _compute_adjaceny_matrix(arr):
    """Compute adjacency matrix for superpixels using GLCM (Grey-Level Co-occurence Matrix)."""

    glcm = greycomatrix(arr, [1], [0, 0.5 * np.pi, np.pi, 1.5 * np.pi],
                        levels=arr.max() + 1)
    glcm = glcm.sum(axis=-1)[..., 0]
    glcm = glcm * (1 - np.eye(glcm.shape[0]))

    return (glcm > 0).astype('uint8')


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
            img = ColorJitter(.3, .3, .3)(img)
            img, mask = _transform_multiple_images(img, mask)

        segments = slic(img, n_segments=int(img.width * img.height / config.SP_AREA),
                        compactness=config.SP_COMPACTNESS)
        adjacency = _compute_adjaceny_matrix(segments)

        img = TF.to_tensor(img)
        segments = torch.LongTensor(segments)
        adjacency = torch.ByteTensor(adjacency)

        if mask is not None:
            mask = np.array(mask)
            mask = np.concatenate([np.expand_dims(mask == i, -1)
                                   for i in range(config.N_CLASSES)], axis=-1)
            mask = torch.LongTensor(mask.astype('int64'))
            return img, segments, adjacency, mask

        return img, segments, adjacency

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
