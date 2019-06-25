import torch

from utils import is_empty_tensor
from utils import empty_tensor
from .config import config


def compute_superpixel_label(mask, segments, sp_idx):
    sp_mask = (mask * (segments == sp_idx).long()).float()
    return sp_mask.sum(dim=(0, 1)) / (sp_mask.sum() + config.epsilon)


def preprocess_superpixels(segments, mask=None):
    """Segment superpixels of a given image and return segment maps and their labels.`
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

    # ordering of superpixels
    sp_idx_list = range(segments.max() + 1)

    if not is_empty_tensor(mask):
        # compute labels for each superpixel
        sp_labels = torch.cat([
            compute_superpixel_label(mask, segments, sp_idx).unsqueeze(0)
            for sp_idx in range(segments.max() + 1)
        ])

        # move labeled superpixels to the front of `sp_idx_list`
        labeled_sps = (sp_labels.sum(dim=-1) > 0).nonzero().squeeze()
        unlabeled_sps = (sp_labels.sum(dim=-1) == 0).nonzero().squeeze()
        sp_idx_list = torch.cat([labeled_sps, unlabeled_sps])

        # quantize superpixel labels (e.g., from (0.7, 0.3) to (1.0, 0.0))
        sp_labels = sp_labels[labeled_sps]
        sp_labels = (sp_labels == sp_labels.max(dim=-1, keepdim=True)[0]).float()
    else:  # no supervision provided
        sp_labels = empty_tensor().to(segments.device)

    # stacking normalized superpixel segment maps
    sp_maps = torch.cat(
        [(segments == sp_idx).unsqueeze(0) for sp_idx in sp_idx_list])
    sp_maps = sp_maps.squeeze().float()
    sp_maps = sp_maps / sp_maps.sum(dim=(1, 2), keepdim=True)

    return sp_maps, sp_labels


def cross_entropy(y_hat, y_true, class_weights=None):
    """Semi-supervised cross entropy loss function.

    Args:
        y_hat: prediction tensor with size (N, C), where C is the number of classes
        y_true: label tensor with size (N, C). A sample won't be counted into loss
            if its label is all zeros.
        class_weights: class weights tensor with size (C,)

    Returns:
        cross_entropy: cross entropy loss computed only on samples with labels
    """

    device = y_hat.device

    # clamp all elements to prevent numerical overflow/underflow
    y_hat = torch.clamp(y_hat, min=config.epsilon, max=(1 - config.epsilon))

    # number of samples with labels
    labeled_samples = torch.sum(y_true.sum(dim=1) > 0).float()

    if labeled_samples.item() == 0:
        return torch.tensor(0.).to(device)

    ce = -y_true * torch.log(y_hat)

    if class_weights is not None:
        ce = ce * class_weights.unsqueeze(0)

    return torch.sum(ce) / labeled_samples


def label_propagate(X, y_l, threshold=0.95):
    """Perform label propagation with similiarity graph.

    Arguments:
        X: input tensor of size (n_l + n_u, d), where n_l is number of labeled samples,
            n_u is number of unlabeled samples and d is the dimension of input
        y_l: label tensor of size (n_l, c), where c is the number of classes
        threshold: similarity threshold for label propagation

    Returns:
        y_u: propagated label tensor of size (n_u, c)
    """

    # disable gradient computation
    X = X.detach()
    y_l = y_l.detach()

    # number of labeled and unlabeled samples
    n_l = y_l.size(0)
    n_u = X.size(0) - n_l

    # compute similarity matrix W
    Xp = X.view(X.size(0), 1, X.size(1))
    W = torch.exp(-torch.einsum('ijk, ijk->ij', X - Xp, X - Xp))

    # sub-matrix of W containing similarities between labeled and unlabeled samples
    W_ul = W[n_l:, :n_l]

    # max_similarities is the maximum similarity for each unlabeled sample
    # src_indexes is the respective labeled sample index
    max_similarities, src_indexes = W_ul.max(dim=1)

    # initialize y_u with zeros
    y_u = torch.zeros(n_u, y_l.size(1)).to(y_l.device)

    # only propagate labels if maximum similarity is above the threhold
    propagated_samples = max_similarities > threshold
    y_u[propagated_samples] = y_l[src_indexes[propagated_samples]]

    return y_u
