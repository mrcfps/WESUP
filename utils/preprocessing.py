"""
Preprocessing utilities.
"""

import os

import numpy as np
from PIL import Image
from skimage.segmentation import slic

import config


def segment_superpixels(img, label=None, cache=None):
    """Segment superpixels of a given image and return segment maps and their labels.

    This function is applicable to three scenarios:

    1. Full annotation: `label` is a mask (`PIL.Image.Image`) with the same height and width as `img`.
        Superpixel maps and their labels will be returned.
    2. Dot annotation: `label` is n_l x 3 array (n_l is the number of points, and each point
        contains information about its x and y coordinates and label). Superpixel maps
        (with n_l labeled superpixels coming first) and n_l labels will be returned.
    3. Inference: `label` is not provided. Only superpixel maps will be returned.
    """

    if cache is not None and os.path.exists(cache):
        data = np.load(cache)
        sp_maps = data['sp_maps']
        sp_maps = sp_maps / sp_maps.sum(axis=(1, 2), keepdims=True)
        if 'sp_labels' in data:
            return sp_maps, data['sp_labels']
        return sp_maps

    img = np.array(img)
    segments = slic(img, n_segments=int(img.shape[0] * img.shape[1] / config.SP_AREA),
                    compactness=config.SP_COMPACTNESS)

    sp_num = segments.max() + 1
    sp_idx_list = range(sp_num)

    if isinstance(label, Image.Image):  # full annotation
        label = np.array(label)
        label = np.concatenate([np.expand_dims(label == i, -1)
                                for i in range(config.N_CLASSES)], axis=-1)
        sp_labels = np.array([
            (label * np.expand_dims(segments == i, -1)
             ).sum(axis=(0, 1)) / np.sum(segments == i)
            for i in range(sp_num)
        ])
        sp_labels = sp_labels == sp_labels.max(axis=-1, keepdims=True)
    elif label is not None:  # dot annotation
        labeled_sps, sp_labels = [], []

        for point in label:
            i, j, class_ = point
            try:
                if segments[i, j] not in labeled_sps:
                    labeled_sps.append(segments[i, j])
                    sp_labels.append(class_)
            except IndexError:
                # point is outside this patch, ignore it
                pass

        # one-hot encoding
        sp_labels = np.array(sp_labels)
        sp_labels = np.concatenate([
            np.expand_dims(sp_labels == i, 0)
            for i in range(config.N_CLASSES)
        ]).T

        unlabeled_sps = list(set(np.unique(segments)) - set(labeled_sps))
        sp_idx_list = labeled_sps + unlabeled_sps

    # stacking normalized superpixel segment maps
    sp_maps = np.concatenate(
        [np.expand_dims(segments == idx, 0) for idx in sp_idx_list])

    if cache is not None:
        data = {'sp_maps': sp_maps}
        if label is not None:
            data['sp_labels'] = sp_labels.astype('uint8')
        np.savez(cache, **data)

    sp_maps = sp_maps / sp_maps.sum(axis=(1, 2), keepdims=True)

    if label is None:
        return sp_maps

    return sp_maps, sp_labels.astype('uint8')
