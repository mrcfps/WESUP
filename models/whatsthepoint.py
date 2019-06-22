import torch
import torch.nn as nn
from torchfcn.models import FCN32s

from utils.data import SegmentationDataset
from utils.data import PointSupervisionDataset
from .base import BaseModel


class WhatsThePointConfig:
    """Configuration for CWDS-MIL model."""

    # Input spatial size.
    input_size = (280, 400)

    # learning rate
    lr = 1e-5

    # momentum for SGD optimizer
    momentum = 0.9

    # weight decay for optimization
    weight_decay = 5e-4

    # numerical stability term
    epsilon = 1e-7


config = WhatsThePointConfig()


class WhatsThePoint(BaseModel):
    """
    What's the Point: Semantic Segmentation with Point Supervision.
    See https://arxiv.org/pdf/1506.02106.pdf.
    """

    def __init__(self, checkpoint=None):
        super().__init__()

        self.fcn = FCN32s(n_class=2)

        if checkpoint is not None:
            self.load_state_dict(checkpoint['model_state_dict'])

        self.summary()

    def get_default_dataset(self, root_dir, train=True):
        if train:
            return PointSupervisionDataset(
                root_dir, target_size=config.input_size, include_obj_prior=True)

        return SegmentationDataset(root_dir, target_size=config.input_size, train=False)

    def get_default_optimizer(self, checkpoint=None):

        def biases():
            """Generator for fetching biases parameters."""
            for layer in self.fcn.children():
                if hasattr(layer, 'bias') and layer.bias is not None:
                    yield layer.bias

        def non_biases():
            """Generator for fetching non-biases parameters."""
            for layer in self.fcn.children():
                if not isinstance(layer, nn.Conv2d):
                    yield from layer.parameters()
                else:
                    yield layer.weight

        optimizer = torch.optim.SGD([
            {'params': non_biases()},
            {'params': biases(), 'lr': 2 * config.lr}
        ], lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay)

        if checkpoint is not None:
            # load previous optimizer states
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        return optimizer, None

    def preprocess(self, *data):
        if self.training:
            img, pixel_mask, point_mask, obj_prior = data
            target_class = (point_mask[..., 1].sum(dim=(1, 2)) > 0).float()
            return img, (pixel_mask, target_class, point_mask, obj_prior)
        else:
            return data

    def forward(self, x):
        return self.fcn(x)[:, 1, ...]

    def compute_loss(self, pred, target, metrics=None):
        """Compute loss for whats-the-point model.

        Args:
            pred: model prediction of size (B, H, W)
            target: a tuple containing following elements:
                1) pixel-level annotation of size (B, H, W, 2)
                2) image-level labels of size (B,) (0 is background and 1 is foreground)
                3) point-level annotation of size (B, H, W, 2)
                4) objectness prior of size (B, H, W)

        Returns:
            loss: sum of image-level, point-level and objectness prior losses
        """

        pred = pred.clamp(min=config.epsilon, max=(1 - config.epsilon))
        _, target_class, point_mask, obj_prior = target

        # image-level loss
        image_pred = pred.max(dim=1).values.max(dim=1).values  # (B,)
        image_loss = (target_class * -torch.log(image_pred) +
                      (1 - target_class) * -torch.log(1 - image_pred))  # (B,)

        # point-level loss
        point_mask = point_mask.float()
        point_loss = (point_mask[..., 0] * -torch.log(1 - pred) +
                      point_mask[..., 1] * -torch.log(pred)).sum(dim=(1, 2)) / point_mask.sum()  # (B,)

        # objectness prior loss
        obj_loss = torch.mean(obj_prior * -torch.log(pred) +
                              (1 - obj_prior) * -torch.log(1 - pred),
                              dim=(1, 2))  # (B,)

        return torch.mean(image_loss + point_loss + obj_loss)

    def postprocess(self, pred, target):
        if self.training:
            target = target[0]
        return pred.round().long(), target.argmax(dim=-1)

    def save_checkpoint(self, ckpt_path, **kwargs):
        """Save model checkpoint."""

        torch.save({
            'model_state_dict': self.state_dict(),
            **kwargs,
        }, ckpt_path)
        print(f'Checkpoint saved to {ckpt_path}.')

    def summary(self):
        print(f'Whats The Point model initialized with input size {config.input_size}')
