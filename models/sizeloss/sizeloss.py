import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.data import SegmentationDataset
from utils.data import AreaConstraintDataset
from utils.data import PointSupervisionDataset
from utils.data import CompoundDataset

from . import networks as networks
from ..base import BaseConfig, BaseTrainer


class SizeLossConfig(BaseConfig):
    """Configuration for CWDS-MIL model."""

    # Network architecture (either 'ENet', 'UNet' or 'ResidualUNet')
    network = 'ResidualUNet'

    # Number of target classes.
    n_classes = 2

    # Area constraint type.
    constraint = 'individual'

    # Input spatial size.
    input_size = (400, 400)

    # Positive constant that weights the importance of constraints.
    lambda_ = 0.01

    # Radius of circle around point annotations
    point_radius = 5

    # Initial learning rate.
    initial_lr = 0.0001

    # numerical stability term
    epsilon = 1e-7


class SizeLoss(nn.Module):
    """Constrained-CNN losses for weakly supervised segmentation.
    https://arxiv.org/abs/1805.04628.
    """

    def __init__(self, **kwargs):
        super().__init__()

        self.kwargs = kwargs
        self.network = getattr(networks, self.kwargs.get('network'))(3, self.kwargs.get('n_classes'))

    def forward(self, x):
        self.network = self.network.to(x.device)
        x = self.network(x)
        x = F.softmax(x, dim=1)
        return x


class SizeLossTrainer(BaseTrainer):

    def __init__(self, model, **kwargs):
        config = SizeLossConfig()
        kwargs = {**config.to_dict(), **kwargs}
        super().__init__(model, **kwargs)

    def get_default_dataset(self, root_dir, train=True, proportion=1.0):
        if train:
            area_dataset = AreaConstraintDataset(root_dir,
                                                 target_size=self.kwargs.get('input_size'),
                                                 area_type='integer',
                                                 constraint=self.kwargs.get('constraint'),
                                                 proportion=proportion)
            point_dataset = PointSupervisionDataset(root_dir,
                                                    target_size=self.kwargs.get('input_size'),
                                                    radius=self.kwargs.get('point_radius'),
                                                    proportion=proportion)
            return CompoundDataset(area_dataset, point_dataset)

        return SegmentationDataset(root_dir, target_size=self.kwargs.get('input_size'), train=False)

    def get_default_optimizer(self):
        optimizer = torch.optim.Adam(self.model.parameters(),
                                     lr=self.kwargs.get('initial_lr'),
                                     betas=(0.9, 0.99))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', patience=100, factor=0.5, min_lr=1e-5, verbose=True)

        return optimizer, scheduler

    def preprocess(self, *data):
        device = self.device
        if self.model.training:
            (img, mask, area), (_, _, point_mask) = data
            point_mask = point_mask.float()
            area = area.float()
            return img.to(device), (mask.to(device), point_mask.to(device), area.to(device))
        else:
            return tuple(datum.to(device) for datum in data)

    def compute_loss(self, pred, target, metrics=None):
        _, point_mask, area = target

        pred = pred.clamp(min=self.kwargs.get('epsilon'), max=(1 - self.kwargs.get('epsilon')))

        # partial cross-entropy
        partial_ce = point_mask * -torch.log(pred) + \
            (1 - point_mask) * -torch.log(1 - pred)  # (B, C, H, W)
        partial_ce = partial_ce.sum(dim=(1, 2, 3))  # (B,)

        # size penalty
        pos_pred = pred[:, 1, ...]  # (B, H, W)
        size_pred = pos_pred.sum(dim=(1, 2))  # (B,)
        size_penalty = (size_pred < area[:, 0]).float() * (size_pred - area[:, 0]) ** 2 + \
            (size_pred > area[:, 1]).float() * (size_pred - area[:, 1]) ** 2  # (B,)

        return torch.mean(partial_ce + self.kwargs.get('lambda_') * size_penalty)

    def postprocess(self, pred, target=None):
        pred = pred.round().long()
        if target is not None:
            if self.model.training:
                target = target[0]
            return pred.argmax(dim=1), target.argmax(dim=1)
        return pred
