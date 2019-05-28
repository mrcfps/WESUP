"""
Data loading utilities.
"""

import csv
import os
import glob
import random
from functools import partial

import numpy as np
from PIL import Image
from skimage.segmentation import slic
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
        images.extend(glob.glob(os.path.join(path, f"*.{ext}")))
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
    """One-shot segmentation dataset."""

    def __init__(self, root_dir, rescale_factor=0.5, train=True):
        # path to original images
        self.img_paths = _list_images(os.path.join(root_dir, "images"))

        # path to mask annotations
        self.mask_paths = None
        if os.path.exists(os.path.join(root_dir, "masks")):
            self.mask_paths = _list_images(os.path.join(root_dir, "masks"))

        # path to point annotations
        self.point_paths = None
        if os.path.exists(os.path.join(root_dir, "points")):
            self.point_paths = sorted(
                glob.glob(os.path.join(root_dir, "points", "*.csv")))

        self.rescale_factor = rescale_factor
        self.train = train

        self.summary()

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx])
        target_height = int(np.ceil(self.rescale_factor * img.height))
        target_width = int(np.ceil(self.rescale_factor * img.width))

        img = img.resize((target_width, target_height), resample=Image.BILINEAR)

        # pixel-level annotation mask
        mask = None
        if self.mask_paths is not None:
            mask = Image.open(self.mask_paths[idx])
            mask = mask.resize((target_width, target_height), resample=Image.NEAREST)

        # point-level annotation mask
        point_mask = None
        if self.point_paths is not None:
            with open(self.point_paths[idx]) as fp:
                points = np.array([[int(d) for d in point]
                                   for point in csv.reader(fp)])
                rescaler = np.array([[self.rescale_factor, self.rescale_factor, 1]])
                points = np.floor(points * rescaler).astype('int')

            # compute point mask
            point_mask = np.zeros((target_height, target_width, config.N_CLASSES), dtype='uint8')
            for i, j, class_ in points:
                point_vec = np.zeros(config.N_CLASSES)
                point_vec[class_] = 1
                point_mask[i, j] = point_vec
            point_mask = Image.fromarray(point_mask)

        if self.train:
            # perform data augmentation
            img = ColorJitter(0.3, 0.3, 0.3)(img)
            img, mask, point_mask = _transform_multiple_images(
                img, mask, point_mask)

        segments = slic(
            img,
            n_segments=int(img.width * img.height / config.SP_AREA),
            compactness=config.SP_COMPACTNESS,
        )

        img = TF.to_tensor(img)
        segments = torch.LongTensor(segments)

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

        if point_mask is not None:
            point_mask = np.array(point_mask, dtype="int64")
            point_mask = torch.LongTensor(point_mask)
        else:
            point_mask = empty_tensor()

        return img, segments, mask, point_mask

    def summary(self):
        """Print summary information."""

        mode = "training" if self.train else "inference"
        print(
            f"Segmentation dataset ({mode}) initialized with {len(self.img_paths)} images.")

        if self.point_paths is not None:
            print("Supervision mode: point-level")
        elif self.mask_paths is not None:
            print("Supervision mode: pixel-level")
        else:
            print("No supervision provided.")


def get_trainval_dataloaders(root_dir, num_workers):
    """Returns training and validation dataloaders."""

    datasets = {
        "train": SegmentationDataset(
            os.path.join(root_dir, "train"), rescale_factor=config.RESCALE_FACTOR
        ),
        "val": SegmentationDataset(
            os.path.join(root_dir, "val"),
            rescale_factor=config.RESCALE_FACTOR,
            train=False,
        ),
    }

    dataloaders = {
        "train": DataLoader(
            datasets["train"], batch_size=1, shuffle=True, num_workers=num_workers
        ),
        "val": DataLoader(
            datasets["val"], batch_size=1, shuffle=True, num_workers=num_workers
        ),
    }

    return dataloaders
