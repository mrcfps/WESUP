import torch


def superpixel_accuracy(sp_pred, sp_labels):
    """Superpixel classification accuracy."""

    return (sp_pred.argmax(dim=-1) == sp_labels).float().mean()


def superpixel_f1(sp_pred, sp_labels):
    """Superpixel classification F1 score."""

    pass


def pixel_accuracy(pred_mask, true_mask):
    """Overall pixel classification accuracy."""

    return (pred_mask == true_mask).float().mean()


def pixel_f1(input, target):
    """Overall pixel classification accuracy."""

    pass

