import torch
import torch.nn.functional as F

from utils.data import SegmentationDataset
from utils.data import AreaConstraintDataset
from utils.data import PointSupervisionDataset
from utils.data import CompoundDataset

from . import networks as networks
from ..base import BaseConfig, BaseModel


class SizeLossConfig(BaseConfig):
    """Configuration for CWDS-MIL model."""

    # Network architecture (either 'ENet', 'UNet' or 'ResidualUNet')
    network = 'ResidualUNet'

    # Number of target classes.
    n_classes = 2

    # Area constraint type.
    constraint = 'individual'

    # Input spatial size.
    input_size = (288, 400)

    # Positive constant that weights the importance of constraints.
    lambda_ = 0.01

    # Radius of circle around point annotations
    point_radius = 5

    # Initial learning rate.
    initial_lr = 0.0001

    # numerical stability term
    epsilon = 1e-7


class SizeLoss(BaseModel):
    """Constrained-CNN losses for weakly supervised segmentation.

    https://arxiv.org/abs/1805.04628.
    """

    def __init__(self, checkpoint=None):
        super().__init__()

        self.config = SizeLossConfig()
        self.network = getattr(networks, self.config.network)(3, self.config.n_classes)

        if checkpoint is not None:
            self.load_state_dict(checkpoint['model_state_dict'])

    def get_default_dataset(self, root_dir, train=True, proportion=1.0):
        if train:
            area_dataset = AreaConstraintDataset(root_dir,
                                                 target_size=self.config.input_size,
                                                 area_type='integer',
                                                 constraint=self.config.constraint,
                                                 proportion=proportion)
            point_dataset = PointSupervisionDataset(root_dir,
                                                    target_size=self.config.input_size,
                                                    radius=self.config.point_radius,
                                                    proportion=proportion)
            return CompoundDataset(area_dataset, point_dataset)

        return SegmentationDataset(root_dir, target_size=self.config.input_size, train=False)

    def get_default_optimizer(self, checkpoint=None):
        optimizer = torch.optim.Adam(self.network.parameters(),
                                     lr=self.config.initial_lr,
                                     betas=(0.9, 0.99))

        if checkpoint is not None:
            # load previous optimizer states
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', patience=10, factor=0.5, min_lr=1e-5, verbose=True)

        return optimizer, None

    def preprocess(self, *data, device='cpu'):
        if self.training:
            (img, mask, area), (_, _, point_mask) = data
            point_mask = point_mask.float()
            area = area.float()
            return img.to(device), (mask.to(device), point_mask.to(device), area.to(device))
        else:
            return tuple(datum.to(device) for datum in data)

    def forward(self, x):
        self.network = self.network.to(x.device)
        x = self.network(x)
        x = F.softmax(x, dim=1)
        return x

    def compute_loss(self, pred, target, metrics=None):
        _, point_mask, area = target

        pred = pred.clamp(min=self.config.epsilon, max=(1 - self.config.epsilon))

        # partial cross-entropy
        partial_ce = point_mask * -torch.log(pred) + \
            (1 - point_mask) * -torch.log(1 - pred)  # (B, C, H, W)
        partial_ce = partial_ce.sum(dim=(1, 2, 3))  # (B,)

        # size penalty
        pos_pred = pred[:, 1, ...]  # (B, H, W)
        size_pred = pos_pred.sum(dim=(1, 2))  # (B,)
        size_penalty = (size_pred < area[:, 0]).float() * (size_pred - area[:, 0]) ** 2 + \
            (size_pred > area[:, 1]).float() * (size_pred - area[:, 1]) ** 2  # (B,)

        return torch.mean(partial_ce + self.config.lambda_ * size_penalty)

    def postprocess(self, pred, target=None):
        pred = pred.round().long()
        if target is not None:
            if self.training:
                target = target[0]
            return pred.argmax(dim=1), target.argmax(dim=1)
        return pred

    def save_checkpoint(self, ckpt_path, **kwargs):
        """Save model checkpoint."""

        torch.save({
            'model_state_dict': self.state_dict(),
            **kwargs,
        }, ckpt_path)
