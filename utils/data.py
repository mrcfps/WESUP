"""
Data loading utilities.
"""

import csv
import os.path as osp
import glob
import random
from functools import partial

import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from torchvision.transforms import ColorJitter

import config
from . import empty_tensor


def _list_images(path):
    """Glob all images within a directory."""

    images = []
    for ext in ("jpg", "jpeg", "png", "bmp"):
        images.extend(glob.glob(osp.join(path, f"*.{ext}")))
    return sorted(images)


def _transform_multiple_images(*imgs):
    """Apply identical transformations to multiple PIL images."""

    def rnd(lower_bound, upper_bound):
        rnd = random.random()
        return lower_bound + (upper_bound - lower_bound) * rnd

    def elastic_transform(image, alpha, sigma, spline_order=1, mode="nearest", random_state=42):
        """Elastic deformation of image as described in [Simard2003]_.
        .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
        Convolutional Neural Networks applied to Visual Document Analysis", in
        Proc. of the International Conference on Document Analysis and
        Recognition, 2003.
        """

        image = np.array(image)

        # don't apply elastic transformation to point mask
        if image.sum() / np.prod(image.shape) < 0.5:
            return Image.fromarray(image)

        np.random.seed(random_state)
        shape = image.shape[:2]

        dx = gaussian_filter((np.random.rand(*shape) * 2 - 1),
                             sigma, mode="constant", cval=0) * alpha
        dy = gaussian_filter((np.random.rand(*shape) * 2 - 1),
                             sigma, mode="constant", cval=0) * alpha

        x, y = np.meshgrid(
            np.arange(shape[0]), np.arange(shape[1]), indexing="ij")
        indices = [np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1))]

        if image.ndim == 2:
            result = map_coordinates(
                image, indices, order=spline_order, mode=mode
            ).reshape(shape)
        else:
            result = np.empty_like(image)
            for i in range(image.shape[2]):
                result[:, :, i] = map_coordinates(
                    image[:, :, i], indices, order=spline_order, mode=mode
                ).reshape(shape)

        return Image.fromarray(result)

    transforms = (
        (1, partial(elastic_transform,
                    alpha=rnd(0, 1000),
                    sigma=rnd(20, 50),
                    random_state=np.random.randint(100))
         ),
        (0.5, TF.hflip),
        (0.5, TF.vflip),
    )

    for prob, transform in transforms:
        if random.random() < prob:
            imgs = [
                transform(img) if img is not None else None for img in imgs]

    if len(imgs) == 1:
        return imgs[0]

    return imgs


class SegmentationDataset(Dataset):
    """Dataset for segmentation task.

    This dataset returns following data when indexing:
        - img: tensor of size (3, H, W) with type float32
        - mask: tensor of size (H, W, C) with type long or an empty tensor
    """

    def __init__(self, root_dir, mode=None, rescale_factor=0.5, train=True):
        # path to original images
        self.img_paths = _list_images(osp.join(root_dir, "images"))

        # path to mask annotations (optional)
        self.mask_paths = None
        if osp.exists(osp.join(root_dir, "masks")):
            self.mask_paths = _list_images(osp.join(root_dir, "masks"))

        self.mode = mode or 'mask' if self.mask_paths is not None else None
        self.rescale_factor = rescale_factor
        self.train = train

        self.summary()

    def __len__(self):
        return len(self.img_paths)

    def _read_image_and_mask(self, idx):
        img = Image.open(self.img_paths[idx])
        target_height = int(np.ceil(self.rescale_factor * img.height))
        target_width = int(np.ceil(self.rescale_factor * img.width))

        img = img.resize((target_width, target_height), resample=Image.BILINEAR)

        # pixel-level annotation mask
        mask = None
        if self.mask_paths is not None:
            mask = Image.open(self.mask_paths[idx])
            mask = mask.resize((target_width, target_height), resample=Image.NEAREST)

        return img, mask

    def _convert_image_and_mask_to_tensor(self, img, mask):
        img = TF.to_tensor(img)
        if mask is not None:
            mask = np.array(mask)
            mask = np.concatenate(
                [np.expand_dims(mask == i, -1)
                 for i in range(config.N_CLASSES)],
                axis=-1,
            )
            mask = torch.LongTensor(mask.astype("int64"))
        else:
            mask = empty_tensor()

        return img, mask

    def __getitem__(self, idx):
        img, mask = self._read_image_and_mask(idx)

        if self.train:
            # perform data augmentation
            img = ColorJitter(0.3, 0.3, 0.3)(img)
            img, mask = _transform_multiple_images(img, mask)

        img, mask = self._convert_image_and_mask_to_tensor(img, mask)

        return img, mask

    def summary(self):
        """Print summary information."""

        print(
            f"Segmentation dataset ({'training' if self.train else 'inference'}) "
            f"initialized with {len(self.img_paths)} images.")

        if self.mode is not None:
            print(f"Supervision mode: {self.mode}")
        else:
            print("No supervision provided.")


class AreaConstraintDataset(SegmentationDataset):
    """Segmentation dataset with area information.

    This dataset returns following data when indexing:
        - img: tensor of size (3, H, W) with type float32
        - mask: tensor of size (H, W, C) with type long or an empty tensor
        - area: a scalar tensor with type float32 or an empty tensor
    """

    def __init__(self, root_dir, rescale_factor=0.5, train=True):
        super().__init__(root_dir, 'area', rescale_factor, train)

        # area information (# foreground pixels divided by total pixels, between 0 and 1)
        self.area_info = pd.read_csv(osp.join(root_dir, "area.csv"),
                                     usecols=['img', 'area'])

    def __getitem__(self, idx):
        img, mask = self._read_image_and_mask(idx)

        if self.train:
            # perform data augmentation
            img = ColorJitter(0.3, 0.3, 0.3)(img)
            img, mask = _transform_multiple_images(img, mask)

        img, mask = self._convert_image_and_mask_to_tensor(img, mask)
        area = torch.tensor(self.area_info.loc[idx]['area'])

        return img, mask, area


class PointSupervisionDataset(SegmentationDataset):
    """One-shot segmentation dataset.

    This dataset returns following data when indexing:
        - img: tensor of size (3, H, W) with type float32
        - pixel_mask: pixel-level annotation of size (H, W, C) with type long or an empty tensor
        - point_mask: point-level annotation of size (H, W, C) with type long or an empty tensor
    """

    def __init__(self, root_dir, point_ratio, rescale_factor=0.5, train=True):
        super().__init__(root_dir, 'point', rescale_factor, train)

        # path to point supervision directory
        self.point_root = osp.join(root_dir, f'points-{point_ratio}')

        if not osp.exists(self.point_root):
            raise Exception(
                f'Point supervision with ratio {point_ratio} does not exist. '
                f'Run scripts/generate_points.py to generate.'
            )

        # path to point annotation files
        self.point_paths = sorted(glob.glob(osp.join(self.point_root, "*.csv")))

    def __getitem__(self, idx):
        img, pixel_mask = self._read_image_and_mask(idx)

        point_mask = None
        if self.point_paths is not None:
            with open(self.point_paths[idx]) as fp:
                points = np.array([[int(d) for d in point]
                                   for point in csv.reader(fp)])
                rescaler = np.array([[self.rescale_factor, self.rescale_factor, 1]])
                points = np.floor(points * rescaler).astype('int')

            # compute point mask
            point_mask = np.zeros((img.height, img.width, config.N_CLASSES), dtype='uint8')
            for i, j, class_ in points:
                point_vec = np.zeros(config.N_CLASSES)
                point_vec[class_] = 1
                point_mask[i, j] = point_vec
            point_mask = Image.fromarray(point_mask)

        if self.train:
            img = ColorJitter(0.3, 0.3, 0.3)(img)
            img, pixel_mask, point_mask = _transform_multiple_images(img, pixel_mask, point_mask)

        img, pixel_mask = self._convert_image_and_mask_to_tensor(img, pixel_mask)

        if point_mask is not None:
            point_mask = np.array(point_mask, dtype="int64")
            point_mask = torch.LongTensor(point_mask)
        else:
            point_mask = empty_tensor()

        return img, pixel_mask, point_mask


def get_trainval_dataloaders(root_dir, mode=None, point_ratio=None, num_workers=4):
    """Returns training and validation dataloaders.

    Args:
        root_dir: path to dataset root
        mode: supervision mode (one of "mask", "point" or "area")
        point_ratio: supervised point ratio (only relevant when `mode` is set to "point")
        num_workers: number of workers for dataloaders

    Returns:
        dataloaders: a dict with two keys "train" and "val" with their respective
            datasets as values
    """

    datasets = {}

    train_dir = osp.join(root_dir, "train")
    if mode == 'point':
        datasets['train'] = PointSupervisionDataset(train_dir,
                                                    point_ratio,
                                                    rescale_factor=config.RESCALE_FACTOR)
    elif mode == 'area':
        datasets['train'] = AreaConstraintDataset(train_dir,
                                                  rescale_factor=config.RESCALE_FACTOR)
    elif mode == 'mask':
        datasets['train'] = SegmentationDataset(train_dir, mode='mask',
                                                rescale_factor=config.RESCALE_FACTOR)

    datasets["val"] = SegmentationDataset(
        osp.join(root_dir, "val"),
        rescale_factor=config.RESCALE_FACTOR,
        train=False,
    )

    dataloaders = {
        "train": DataLoader(
            datasets["train"], batch_size=1, shuffle=True, num_workers=num_workers
        ),
        "val": DataLoader(
            datasets["val"], batch_size=1, shuffle=True, num_workers=num_workers
        ),
    }

    return dataloaders
