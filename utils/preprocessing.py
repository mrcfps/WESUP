"""
Preprocessing utilities.
"""

import os
from zipfile import BadZipFile

import numpy as np
from PIL import Image
from skimage.segmentation import slic

import config


def _compute_superpixel_label(mask, segments, sp_idx):
    sp_mask = mask * np.expand_dims(segments == sp_idx, -1)
    if sp_mask.sum() == 0:
        return np.array([0, 0])
    return sp_mask.sum(axis=(0, 1)) / sp_mask.sum()


def segment_superpixels(img, mask=None):
    """Segment superpixels of a given image and return segment maps and their labels.

    Arguments:
        img: image with shape (H, W, n_channels)
        mask (optional): annotation mask with shape (H, W, C). Each pixel is a one-hot encoded
            label vector. If this vector is all zeros, then its class is unknown.

    Returns
        sp_maps: superpixel maps with shape (n_superpixles, H, W)
        sp_labels: superpixel labels with shape (n_labels, C), only when `mask` is given.
            `n_labels` could be smaller than `n_superpixels` in the case of point supervision,
            where `sp_labels` correspond to labels of the first `n_labels` superpixels.
    """

    segments = slic(img, n_segments=int(img.shape[0] * img.shape[1] / config.SP_AREA),
                    compactness=config.SP_COMPACTNESS)

    # ordering of superpixels
    sp_idx_list = range(segments.max() + 1)

    if mask is not None:
        sp_labels = np.array([
            _compute_superpixel_label(mask, segments, sp_idx)
            for sp_idx in range(segments.max() + 1)
        ])

        # move labeled superpixels to the front of `sp_idx_list`
        labeled_sps = np.where(sp_labels.sum(axis=-1) > 0)[0]
        unlabeled_sps = np.where(sp_labels.sum(axis=-1) == 0)[0]
        sp_idx_list = np.r_[labeled_sps, unlabeled_sps]

        # quantize superpixel labels (e.g. from (0.7, 0.3) to (1.0, 0.0))
        sp_labels = sp_labels[labeled_sps]
        sp_labels = sp_labels == sp_labels.max(axis=-1, keepdims=True)

    # stacking normalized superpixel segment maps
    sp_maps = np.concatenate(
        [np.expand_dims(segments == idx, 0) for idx in sp_idx_list])
    sp_maps = sp_maps / sp_maps.sum(axis=(1, 2), keepdims=True)
    sp_maps = sp_maps.astype('float32')

    if mask is None:
        return sp_maps

    return sp_maps, sp_labels.astype('float32')
