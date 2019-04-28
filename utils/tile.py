"""
Utilities for (non-)overlapping-tile (or sliding window) predictions.
"""

import numpy as np


def compute_patches_grid_shape(img, patch_size, stride):
    """Calculate height and width of patches grid needed to predict an image."""

    height, width = img.shape[:2]
    n_h = np.ceil((height - patch_size) / stride + 1)
    n_w = np.ceil((width - patch_size) / stride + 1)

    return int(n_h), int(n_w)

def compute_padded_image_size(img, patch_size, stride):
    """Calculate padded height and width."""

    n_h, n_w = compute_patches_grid_shape(img, patch_size, stride)
    ext_h = patch_size + stride * (n_h - 1)
    ext_w = patch_size + stride * (n_w - 1)

    return ext_h, ext_w


def pad_image(img, patch_size, stride):
    """Pad the image with zeros so that sliding windows won't move out."""

    height, width, n_channels = img.shape
    ext_height, ext_width = compute_padded_image_size(img, patch_size, stride)
    ext_img = np.zeros((ext_height, ext_width, n_channels), dtype='uint8')
    ext_img[:height, :width] = img

    return ext_img


def divide_image_to_patches(img, patch_size, stride=None, pad=True):
    """
    Divide a large image (mask) to patches with (possibly overlapping) tiling window strategy.
    """

    stride = stride or patch_size
    if not 0 < stride <= patch_size:
        raise ValueError(
            'stride should be positive and smaller than or equal to patch_size')

    if len(img.shape) == 2:  # this is a mask
        img = np.expand_dims(img, -1)

    height, width = img.shape[:2]

    if pad:
        img = pad_image(img, patch_size, stride)
        height, width = img.shape[:2]

    x = []

    for i in range(0, height - patch_size + 1, stride):
        for j in range(0, width - patch_size + 1, stride):
            x.append(img[i:i + patch_size, j:j + patch_size, :])

    return np.array(x).astype('uint8')


def combine_patches_to_image(patches, grid_shape, target_size=None, stride=None):
    """Combine patches back to a single image (mask).

    Arguments:
        patches: numpy array with shape (n_patches, patch_size, patch_size, n_channels)
        grid_shape: a tuple containing number of patches along height and width respectively
        target_size: a tuple containing target height and width of combined image
        stride: stride to use when combining patches

    Returns:
        img: combined image
    """

    n_h, n_w = grid_shape
    if len(patches) != n_h * n_w:
        raise ValueError(
            f'{len(patches)} patches cannot be combined with grid shape {grid_shape}'
        )

    patch_size = patches.shape[1]
    stride = stride or patch_size
    height = patch_size + stride * (n_h - 1)
    width = patch_size + stride * (n_w - 1)

    # The last channel is the number of overlapping patches for a given pixel,
    # used for averaging predictions from multiple windows.
    combined = np.zeros((height, width, patches.shape[-1] + 1))

    # counter for patch index
    counter = 0

    for i in range(0, height - patch_size + 1, stride):
        for j in range(0, width - patch_size + 1, stride):
            patch = combined[i:i + patch_size, j:j + patch_size, :-1]
            overlaps = combined[i:i + patch_size, j:j + patch_size, -1:]
            patch = (patch * overlaps + patches[counter]) / (overlaps + 1)
            combined[i:i + patch_size, j:j + patch_size, :-1] = patch
            overlaps += 1.
            counter += 1

    if target_size:
        height, width = target_size

    return combined[:height, :width, :-1]
