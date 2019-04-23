"""
Utilities for (non-)overlapping-tile (or sliding window) predictions.
"""

import numpy as np


def _get_extended_image_size(height, width, patch_size, stride):
    """Calculate extended height and width for given stride."""

    ext_height, ext_width = 0, 0

    def sliding_distance(n_windows, window_size, stride):
        return window_size * n_windows - (window_size - stride) * (n_windows - 1)

    if height < patch_size:
        ext_height = patch_size
    else:
        for n in range(height):
            distance = sliding_distance(n, patch_size, stride)
            if distance > height:
                ext_height = distance
                break

    if width < patch_size:
        ext_width = patch_size
    else:
        for n in range(width):
            distance = sliding_distance(n, patch_size, stride)
            if distance > width:
                ext_width = distance
                break

    return ext_height, ext_width


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

    height, width, n_channels = img.shape

    if pad:
        # extend the original image so that the sliding window won't move out of the image
        ext_height, ext_width = _get_extended_image_size(
            height, width, patch_size, stride)
        ext_img = np.zeros((ext_height, ext_width, n_channels))
        ext_img[:height, :width] = img

        # overwrite original values
        img = ext_img
        height, width = ext_height, ext_width

    x = []

    for i in range(0, height - patch_size + 1, stride):
        for j in range(0, width - patch_size + 1, stride):
            x.append(img[i:i + patch_size, j:j + patch_size, :])

    return np.array(x).astype('uint8')


def combine_patches_to_image(y_pred, img, stride):
    """Combine patches back to a single image (mask)."""

    counter = 0
    height, width = img.shape[:2]
    output_size = y_pred.shape[1]

    # The last channel is the number of overlapping patches for a given pixel,
    # used for averaging predictions from multiple windows.
    combined = np.zeros((height, width, y_pred.shape[-1] + 1))

    for i in range(0, height - output_size + 1, stride):
        for j in range(0, width - output_size + 1, stride):
            patch = combined[i:i + output_size, j:j + output_size, :-1]
            overlaps = combined[i:i + output_size, j:j + output_size, -1:]
            patch = (patch * overlaps + y_pred[counter]) / (overlaps + 1)
            combined[i:i + output_size, j:j + output_size, :-1] = patch
            overlaps += 1.
            counter += 1

    return combined[:height, :width, :-1]
