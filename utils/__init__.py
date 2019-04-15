import torch


def predict_whole_patch(sp_pred, sp_labels, sp_maps):
    """
    Calculate patch prediction from superpixel predictions and labels.
    """

    sp_pred = sp_pred.argmax(dim=-1)

    # flatten sp_maps to one channel
    sp_maps = sp_maps.argmax(dim=0)

    # initialize prediction mask
    pred_mask = torch.zeros_like(sp_maps).to(sp_maps.device)

    for n in range(sp_maps.max().item() + 1):
        pred_mask[sp_maps == n] = sp_pred[n]

    return pred_mask
