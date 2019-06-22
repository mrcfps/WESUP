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
from skimage.transform import resize
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

import torch
from torch.utils.data import Dataset
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

    def __init__(self, root_dir, mode=None, target_size=None,
                 rescale_factor=None, train=True):
        """Initialize a new SegmentationDataset.

        Args:
            root_dir: path to dataset root
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
        img = Image.open(self.img_paths[idx])
        mask = None
        if self.mask_paths is not None:
            mask = Image.open(self.mask_paths[idx])
        img, mask = self._resize_image_and_mask(img, mask)

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

    def __init__(self, root_dir, target_size=None, rescale_factor=None, train=True):
        super().__init__(root_dir, 'area', target_size, rescale_factor, train)

        # area information (# foreground pixels divided by total pixels, between 0 and 1)
        self.area_info = pd.read_csv(osp.join(root_dir, "area.csv"),
                                     usecols=['img', 'area'])

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx])

        mask = None
        if self.mask_paths is not None:
            mask = Image.open(self.mask_paths[idx])
        img, mask = self._resize_image_and_mask(img, mask)

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
        - obj_prior (optional): objectness prior of size (H, W) with type float32
    """

    def __init__(self, root_dir, target_size=None, rescale_factor=None,
                 include_obj_prior=False, train=True):
        super().__init__(root_dir, 'point', target_size, rescale_factor, train)

        # path to point supervision directory
        self.point_root = osp.join(root_dir, f'points')

        # path to point annotation files
        self.point_paths = sorted(glob.glob(osp.join(self.point_root, "*.csv")))

        # path to objectness prior
        self.obj_prior_paths = sorted(glob.glob(
            osp.join(root_dir, 'objectness', '*.npy'))) if include_obj_prior else None

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx])

        pixel_mask = None
        if self.mask_paths is not None:
            pixel_mask = Image.open(self.mask_paths[idx])

        orig_height, orig_width = img.height, img.width
        img, pixel_mask = self._resize_image_and_mask(img, pixel_mask)

        point_mask = None

        # how much we would like to rescale coordinates of each point
        # (the last dimension is target class, which should be kept the same)
        if self.rescale_factor is None:
            rescaler = np.array([
                [self.target_size[0] / orig_height, self.target_size[1] / orig_width, 1]
            ])
        else:
            rescaler = np.array([[self.rescale_factor, self.rescale_factor, 1]])

        if self.point_paths is not None:
            with open(self.point_paths[idx]) as fp:
                points = np.array([[int(d) for d in point]
                                   for point in csv.reader(fp)])
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

        if self.obj_prior_paths is not None:
            # obj_prior = TF.to_tensor(Image.open(self.obj_prior_paths[idx])).squeeze()
            obj_prior = np.load(self.obj_prior_paths[idx])
            obj_prior = resize(obj_prior, self.target_size)
            obj_prior = torch.Tensor(obj_prior)
            return img, pixel_mask, point_mask, obj_prior

        return img, pixel_mask, point_mask
