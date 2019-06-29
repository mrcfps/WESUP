"""
Data loading utilities.
"""

import csv
import os.path as osp
import glob

import numpy as np
import pandas as pd
from PIL import Image
import albumentations as A

import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

from . import empty_tensor


def _list_images(path):
    """Glob all images within a directory."""

    images = []
    for ext in ("jpg", "jpeg", "png", "bmp"):
        images.extend(glob.glob(osp.join(path, f"*.{ext}")))
    return sorted(images)


class SegmentationDataset(Dataset):
    """Dataset for segmentation task.

    This dataset returns following data when indexing:
        - img: tensor of size (3, H, W) with type float32
        - mask: tensor of size (C, H, W) with type long or an empty tensor
    """

    def __init__(self, root_dir, mode=None, target_size=None,
                 rescale_factor=None, train=True, n_classes=2):
        """Initialize a new SegmentationDataset.

        Args:
            root_dir: path to dataset root
            n_classes: number of target classes
            mode: one of `mask`, `area` or `point`
            target_size: desired output spatial size
            rescale_factor: multiplier for spatial size
            train: whether in training mode
        """

        # path to original images
        self.img_paths = _list_images(osp.join(root_dir, "images"))

        # path to mask annotations (optional)
        self.mask_paths = None
        if osp.exists(osp.join(root_dir, "masks")):
            self.mask_paths = _list_images(osp.join(root_dir, "masks"))

        self.mode = mode or 'mask' if self.mask_paths is not None else None
        self.target_size = target_size
        self.rescale_factor = rescale_factor
        self.train = train
        self.n_classes = n_classes

        self.summary()

    def __len__(self):
        return len(self.img_paths)

    def _resize_image_and_mask(self, img, mask=None):
        if self.target_size is not None:
            target_height, target_width = self.target_size
        elif self.rescale_factor is not None:
            target_height = int(np.ceil(self.rescale_factor * img.height))
            target_width = int(np.ceil(self.rescale_factor * img.width))
        else:
            target_height, target_width = img.height, img.width

        img = img.resize((target_width, target_height), resample=Image.BILINEAR)

        # pixel-level annotation mask
        if mask is not None:
            mask = mask.resize((target_width, target_height), resample=Image.NEAREST)

        return np.array(img), np.array(mask)

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
            mask = np.array(mask)
            mask = np.concatenate(
                [np.expand_dims(mask == i, 0)
                 for i in range(self.n_classes)])
            mask = torch.LongTensor(mask.astype("int64"))
        else:
            mask = empty_tensor()

        return img, mask

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx])
        mask = None
        if self.mask_paths is not None:
            mask = Image.open(self.mask_paths[idx])
        img, mask = self._resize_image_and_mask(img, mask)

        if self.train:
            img, mask = self._augment(img, mask)

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
        - mask: tensor of size (C, H, W) with type long or an empty tensor
        - area: a scalar tensor with type float32 or an empty tensor
    """

    def __init__(self, root_dir, target_size=None, rescale_factor=None, train=True):
        super().__init__(root_dir, 'area', target_size, rescale_factor, train)

        # area information (# foreground pixels divided by total pixels, between 0 and 1)
        self.area_info = pd.read_csv(osp.join(root_dir, "area.csv"),
                                     usecols=['img', 'area'])

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
        img = Image.open(self.img_paths[idx])

        mask = None
        if self.mask_paths is not None:
            mask = Image.open(self.mask_paths[idx])
        img, mask = self._resize_image_and_mask(img, mask)

        if self.train:
            img, mask = self._augment(img, mask)

        img, mask = self._convert_image_and_mask_to_tensor(img, mask)
        area = torch.tensor(self.area_info.loc[idx]['area'])

        return img, mask, area


class PointSupervisionDataset(SegmentationDataset):
    """One-shot segmentation dataset.

    This dataset returns following data when indexing:
        - img: tensor of size (3, H, W) with type float32
        - pixel_mask: pixel-level annotation of size (C, H, W) with type long or an empty tensor
        - point_mask: point-level annotation of size (C, H, W) with type long or an empty tensor
    """

    def __init__(self, root_dir, target_size=None, rescale_factor=None, train=True):
        super().__init__(root_dir, 'point', target_size, rescale_factor, train)

        # path to point supervision directory
        self.point_root = osp.join(root_dir, f'points')

        # path to point annotation files
        self.point_paths = sorted(glob.glob(osp.join(self.point_root, "*.csv")))

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
        img = Image.open(self.img_paths[idx])

        pixel_mask = None
        if self.mask_paths is not None:
            pixel_mask = Image.open(self.mask_paths[idx])

        orig_height, orig_width = img.height, img.width
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
        with open(self.point_paths[idx]) as fp:
            points = np.array([[int(d) for d in point]
                               for point in csv.reader(fp)])
            points = np.floor(points * rescaler).astype('int')

        if self.train:
            img, pixel_mask, points = self._augment(img, pixel_mask, points)

        point_mask = np.zeros((self.n_classes, *img.shape[:2]), dtype='uint8')
        for x, y, class_ in points:
            point_vec = np.zeros(self.n_classes)
            point_vec[class_] = 1
            point_mask[:, y, x] = point_vec

        img, pixel_mask = self._convert_image_and_mask_to_tensor(img, pixel_mask)

        if point_mask is not None:
            point_mask = torch.LongTensor(point_mask.astype('int64'))
        else:
            point_mask = empty_tensor()

        return img, pixel_mask, point_mask
