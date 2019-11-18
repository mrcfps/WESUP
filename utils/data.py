"""
Data loading utilities.
"""

import csv
from functools import partial
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import albumentations as A
from skimage.io import imread
from skimage.segmentation import find_boundaries
from skimage.morphology import dilation
from skimage.transform import resize

import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

from . import empty_tensor

resize_mask = partial(resize, order=0, preserve_range=True, anti_aliasing=False)


def resize_img(img, target_size):
    img = resize(img, target_size, order=1, anti_aliasing=False)
    return (img * 255).astype('uint8')


class SegmentationDataset(Dataset):
    """Dataset for segmentation task.

    This dataset returns following data when indexing:
        - img: tensor of size (3, H, W) with type float32
        - mask: tensor of size (C, H, W) with type long or an empty tensor
        - cont (optional): tensor of size (C, H, W) with type long, only when `contour` is `True`
    """

    def __init__(self, root_dir, mode=None, contour=False, target_size=None, rescale_factor=None,
                 multiscale_range=None, train=True, proportion=1, n_classes=2, seed=0):
        """Initialize a new SegmentationDataset.

        Args:
            root_dir: path to dataset root
            mode: one of `mask`, `area` or `point`
            contour: whether to include contours
            target_size: desired output spatial size
            rescale_factor: multiplier for spatial size
            multiscale_range: a tuple containing the limits of random rescaling
            train: whether in training mode
            proportion: proportion of data to be used (between 0 and 1) 
            n_classes: number of target classes
            seed: random seed
        """

        self.root_dir = Path(root_dir).expanduser()

        # path to original images
        self.img_paths = sorted((self.root_dir / 'images').iterdir())

        # path to mask annotations (optional)
        self.mask_paths = None
        if (self.root_dir / 'masks').exists():
            self.mask_paths = sorted((self.root_dir / 'masks').iterdir())

        self.mode = mode or 'mask' if self.mask_paths is not None else None

        if self.mode != 'mask' and contour:
            raise ValueError('mask is required for providing contours')

        self.contour = contour
        self.target_size = target_size
        self.rescale_factor = rescale_factor

        self.train = train
        self.proportion = proportion
        self.n_classes = n_classes
        self.multiscale_range = multiscale_range

        # indexes to pick image/mask from
        self.picked = np.arange(len(self.img_paths))
        if self.proportion < 1:
            np.random.seed(seed)
            np.random.shuffle(self.picked)
            self.picked = self.picked[:len(self)]
            self.picked.sort()

    def __len__(self):
        return int(self.proportion * len(self.img_paths))

    def _resize_image_and_mask(self, img, mask=None):
        height, width = img.shape[:2]
        if self.target_size is not None:
            target_height, target_width = self.target_size
        elif self.multiscale_range is not None:
            self.rescale_factor = np.random.uniform(*self.multiscale_range)
            target_height = int(np.ceil(self.rescale_factor * height))
            target_width = int(np.ceil(self.rescale_factor * width))
        elif self.rescale_factor is not None:
            target_height = int(np.ceil(self.rescale_factor * height))
            target_width = int(np.ceil(self.rescale_factor * width))
        else:
            target_height, target_width = height, width

        img = resize_img(img, (target_height, target_width))

        # pixel-level annotation mask
        if mask is not None:
            mask = resize_mask(mask, (target_height, target_width))

        return img, mask

    def _augment(self, *data):
        img, mask = data

        transformer = A.Compose([
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10,
                                 val_shift_limit=10, p=1),
            A.RandomBrightnessContrast(brightness_limit=0.1,
                                       contrast_limit=0.1, p=1),
            A.CLAHE(p=0.5),
            A.ElasticTransform(p=0.5),
            A.Blur(blur_limit=3, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(p=0.8),
        ])
        augmented = transformer(image=img, mask=mask)

        return augmented['image'], augmented.get('mask', None)

    def _convert_image_and_mask_to_tensor(self, img, mask):
        img = TF.to_tensor(img)
        if mask is not None:
            if self.contour:
                cont = dilation(find_boundaries(mask))
            mask = np.concatenate([np.expand_dims(mask == i, 0)
                                   for i in range(self.n_classes)])
            mask = torch.as_tensor(mask.astype('int64'), dtype=torch.long)
        else:
            mask = empty_tensor()

        if self.contour:
            cont = np.concatenate([np.expand_dims(cont == i, 0)
                                   for i in range(self.n_classes)])
            cont = torch.as_tensor(cont.astype('int64'), dtype=torch.long)
            return img, mask, cont

        return img, mask

    def __getitem__(self, idx):
        idx = self.picked[idx]
        img = imread(str(self.img_paths[idx]))
        mask = None
        if self.mask_paths is not None:
            mask = imread(str(self.mask_paths[idx]))
        img, mask = self._resize_image_and_mask(img, mask)

        if self.train:
            img, mask = self._augment(img, mask)

        return self._convert_image_and_mask_to_tensor(img, mask)

    def summary(self, logger=None):
        """Print summary information."""

        lines = [
            f"Segmentation dataset ({'training' if self.train else 'inference'}) ",
            f"initialized with {len(self)} images from {self.root_dir}.",
        ]

        if self.mode is not None:
            lines.append(f"Supervision mode: {self.mode}")
        else:
            lines.append("No supervision provided.")

        lines = '\n'.join(lines)

        if logger is not None:
            logger.info(lines)
        else:
            print(lines)


class AreaConstraintDataset(SegmentationDataset):
    """Segmentation dataset with area information.

    This dataset returns following data when indexing:
        - img: tensor of size (3, H, W) with type float32
        - mask: tensor of size (C, H, W) with type long or an empty tensor
        - area: a 2-element (lower and upper bound) vector tensor with type float32
    """

    def __init__(self, root_dir, target_size=None, rescale_factor=None, area_type='decimal',
                 constraint='equality', margin=0.1, train=True, proportion=1.0):
        """Construct a new AreaConstraintDataset instance.

        Args:
            root_dir: path to dataset root
            target_size: desired output spatial size
            rescale_factor: multiplier for spatial size
            area_type: either 'decimal' (relative size) or 'integer' (total number of positive pixels)
            constraint: either 'equality' (equality area constraint), 'common' (inequality
                common bound constraint) or 'individual' (inequality individual bound constraint)
            margin: soft margin of inequality constraint, only relevant when `constraint` is
                set to 'individual'
            train: whether in training mode
            proportion: proportion of data to be used (between 0 and 1)

        Returns:
            dataset: a new AreaConstraintDataset instance
        """
        super().__init__(root_dir, mode='area', target_size=target_size,
                         rescale_factor=rescale_factor, train=train, proportion=proportion)

        # area information (# foreground pixels divided by total pixels, between 0 and 1)
        self.area_info = pd.read_csv(self.root_dir / 'area.csv', usecols=['img', 'area'])

        self.area_type = area_type
        self.constraint = constraint
        self.margin = margin

    def _augment(self, *data):
        img, mask = data

        transformer = A.Compose([
            A.HueSaturationValue(p=1),
            A.RandomBrightnessContrast(brightness_limit=0.3,
                                       contrast_limit=0.3, p=1),
            A.CLAHE(p=0.5),
            A.ElasticTransform(p=0.5),
            A.Blur(blur_limit=3, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
        ])
        augmented = transformer(image=img, mask=mask)

        return augmented['image'], augmented.get('mask', None)

    def __getitem__(self, idx):
        idx = self.picked[idx]
        img = imread(str(self.img_paths[idx]))

        mask = None
        if self.mask_paths is not None:
            mask = imread(str(self.mask_paths[idx]))
        img, mask = self._resize_image_and_mask(img, mask)

        if self.train:
            img, mask = self._augment(img, mask)

        img, mask = self._convert_image_and_mask_to_tensor(img, mask)

        if self.area_type == 'decimal':
            area = self.area_info.loc[idx]['area']
        else:  # integer
            area = mask[1].sum().float()

        if self.constraint == 'equality':
            area = torch.tensor([area, area])
        elif self.constraint == 'individual':
            area = torch.tensor([area * (1 - self.margin), area * (1 + self.margin)]).long()
        else:  # common
            lower = self.area_info.area.min()
            upper = self.area_info.area.max()
            if self.area_type == 'integer':
                lower = int(lower * np.prod(self.target_size))
                upper = int(upper * np.prod(self.target_size))
            area = torch.tensor([lower, upper])

        return img, mask, area


class PointSupervisionDataset(SegmentationDataset):
    """One-shot segmentation dataset.

    This dataset returns following data when indexing:
        - img: tensor of size (3, H, W) with type float32
        - pixel_mask: pixel-level annotation of size (C, H, W) with type long or an empty tensor
        - point_mask: point-level annotation of size (C, H, W) with type long or an empty tensor
    """

    def __init__(self, root_dir, target_size=None, rescale_factor=None,
                 multiscale_range=None, radius=0, train=True, proportion=1):
        super().__init__(root_dir, mode='point', target_size=target_size,
                         rescale_factor=rescale_factor, train=train,
                         proportion=proportion, multiscale_range=multiscale_range)

        # path to point supervision directory
        self.point_root = self.root_dir / 'points'

        # path to point annotation files
        self.point_paths = sorted(self.point_root.glob('*.csv'))

        self.radius = radius

    def _augment(self, *data):
        img, mask, points = data

        # transforms applied to images and masks
        appearance_transformer = A.Compose([
            A.HueSaturationValue(p=1),
            A.RandomBrightnessContrast(brightness_limit=0.3,
                                       contrast_limit=0.3, p=1),
            A.CLAHE(p=0.5),
            A.Blur(blur_limit=3, p=0.5),
        ])

        # transforms applied to images, masks and points
        position_transformer = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(p=1),
        ], keypoint_params={'format': 'xy'})

        augmented = appearance_transformer(image=img, mask=mask)
        temp_img, temp_mask = augmented['image'], augmented['mask']

        augmented = position_transformer(image=temp_img, mask=temp_mask, keypoints=points)

        return augmented['image'], augmented.get('mask', None), augmented.get('keypoints', None)

    def __getitem__(self, idx):
        idx = self.picked[idx]
        img = imread(str(self.img_paths[idx]))

        pixel_mask = None
        if self.mask_paths is not None:
            pixel_mask = imread(str(self.mask_paths[idx]))

        orig_height, orig_width = img.shape[:2]
        img, pixel_mask = self._resize_image_and_mask(img, pixel_mask)

        # how much we would like to rescale coordinates of each point
        # (the last dimension is target class, which should be kept the same)
        if self.rescale_factor is None:
            rescaler = np.array([
                [self.target_size[1] / orig_width, self.target_size[0] / orig_height, 1]
            ])
        else:
            rescaler = np.array([[self.rescale_factor, self.rescale_factor, 1]])

        # read points from csv file
        with open(str(self.point_paths[idx])) as fp:
            points = np.array([[int(d) for d in point]
                               for point in csv.reader(fp)])
            points = np.floor(points * rescaler).astype('int')

        if self.train:
            img, pixel_mask, points = self._augment(img, pixel_mask, points)

        point_mask = np.zeros((self.n_classes, *img.shape[:2]), dtype='uint8')
        for x, y, class_ in points:
            cv2.circle(point_mask[class_], (x, y), self.radius, 1, -1)

        if point_mask is not None:
            point_mask = torch.as_tensor(point_mask.astype('int64'), dtype=torch.long)
        else:
            point_mask = empty_tensor()

        img, pixel_mask = self._convert_image_and_mask_to_tensor(img, pixel_mask)

        return img, pixel_mask, point_mask


class WESUPV2Dataset(SegmentationDataset):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if (self.root_dir / 'spl-masks').exists():
            self.mask_paths = sorted((self.root_dir / 'spl-masks').iterdir())

    def _generate_coords(self, shape):
        x = np.linspace(0, 1, shape[0])
        y = np.linspace(0, 1, shape[1])
        coords = torch.as_tensor([np.tile(x, len(y)), np.repeat(y, len(x))],
                                 dtype=torch.float32)

        return coords.view(2, shape[0], shape[1])

    def __getitem__(self, idx):
        idx = self.picked[idx]
        img = imread(str(self.img_paths[idx]))
        mask = None
        if self.mask_paths is not None:
            mask = np.load(self.mask_paths[idx])
        img, mask = self._resize_image_and_mask(img, mask)

        if self.train:
            img, mask = self._augment(img, mask)

        coords = self._generate_coords(img.shape)
        img = TF.to_tensor(img)
        mask = torch.as_tensor(mask.transpose(2, 0, 1), dtype=torch.long)

        return img, mask, coords


class CompoundDataset(Dataset):

    def __init__(self, *datasets):
        self.datasets = datasets

    def __len__(self):
        return len(self.datasets[0])

    def __getitem__(self, idx):
        return tuple(dataset[idx] for dataset in self.datasets)

    def summary(self, logger=None):
        for dataset in self.datasets:
            dataset.summary(logger=logger)
