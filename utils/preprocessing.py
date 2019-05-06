"""
Preprocessing utilities.
"""

import torch

import config


def _compute_superpixel_label(mask, segments, sp_idx):
    sp_mask = (mask * (segments == sp_idx).long()).float()
    return sp_mask.sum(dim=(0, 1)) / (sp_mask.sum() + config.EPSILON)


def preprocess_superpixels(segments, mask=None):
    """Segment superpixels of a given image and return segment maps and their labels.

    Arguments:
        segments: slic segments tensor with shape (H, W)
        mask (optional): annotation mask tensor with shape (H, W, C). Each pixel is a one-hot
            encoded label vector. If this vector is all zeros, then its class is unknown.

    Returns
        sp_maps: superpixel maps with shape (n_superpixels, H, W)
        sp_labels: superpixel labels with shape (n_labels, C), only when `mask` is given.
            `n_labels` could be smaller than `n_superpixels` in the case of point supervision,
            where `sp_labels` correspond to labels of the first `n_labels` superpixels.
    """

    segments.unsqueeze_(-1)

    # ordering of superpixels
    sp_idx_list = range(segments.max() + 1)

    if mask is not None:
        sp_labels = torch.cat([
            _compute_superpixel_label(mask, segments, sp_idx).unsqueeze(0)
            for sp_idx in range(segments.max() + 1)
        ])

        # move labeled superpixels to the front of `sp_idx_list`
        labeled_sps = (sp_labels.sum(dim=-1) > 0).nonzero().squeeze()
        unlabeled_sps = (sp_labels.sum(dim=-1) == 0).nonzero().squeeze()
        sp_idx_list = torch.cat([labeled_sps, unlabeled_sps])

        # quantize superpixel labels (e.g. from (0.7, 0.3) to (1.0, 0.0))
        sp_labels = sp_labels[labeled_sps]
        sp_labels = (sp_labels == sp_labels.max(dim=-1, keepdim=True)[0]).float()

    # stacking normalized superpixel segment maps
    sp_maps = torch.cat(
        [(segments == idx).unsqueeze(0) for idx in sp_idx_list])
    sp_maps = sp_maps.squeeze().float()
    sp_maps = sp_maps / sp_maps.sum(dim=(1, 2), keepdim=True)

    if mask is None:
        return sp_maps

    return sp_maps, sp_labels
