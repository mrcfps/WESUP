import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.models import vgg16

from utils.data import SegmentationDataset
from utils.data import AreaConstraintDataset
from .base import BaseConfig, BaseTrainer


class CDWSConfig(BaseConfig):
    """Configuration for CWDS-MIL model."""

    # Input spatial size.
    input_size = (400, 400)

    # Fixed fusion weights
    fusion_weights = (0.2, 0.35, 0.45)

    # Weights of area constrains loss
    side_ac_weights = (2.5, 5, 10)
    fuse_ac_weight = 10

    # learning rates
    vgg_lr = 1e-3
    side_lr = 1e-5

    # weight decay for optimization
    weight_decay = 5e-4

    # Generalized mean parameter
    gmp = 4

    # numerical stability term
    epsilon = 1e-7


class CDWS(nn.Module):
    """
    Constrained deep weak supervision network for histopathology image segmentation.
    See https://arxiv.org/pdf/1701.00794.pdf.
    """

    def __init__(self, **kwargs):
        """Initialize CDWS-MIL.

        Kwargs:
            fusion_weights: weights for fusing multiple outputs

        Returns:
            model: a new CDWS model
        """

        super().__init__()

        self.kwargs = kwargs

        self.vgg = nn.ModuleDict({
            # First stage
            'conv1_1': nn.Conv2d(3, 64, 3, padding=1),
            'conv1_2': nn.Conv2d(64, 64, 3, padding=1),
            'pool1': nn.MaxPool2d(2, stride=2),

            # Second stage
            'conv2_1': nn.Conv2d(64, 128, 3, padding=1),
            'conv2_2': nn.Conv2d(128, 128, 3, padding=1),
            'pool2': nn.MaxPool2d(2, stride=2),

            # Third stage
            'conv3_1': nn.Conv2d(128, 256, 3, padding=1),
            'conv3_2': nn.Conv2d(256, 256, 3, padding=1),
            'conv3_3': nn.Conv2d(256, 256, 3, padding=1),
        })

        # Side output conv layers.
        self.side = nn.ModuleDict({
            'side_conv1': nn.Conv2d(64, 1, (1, 1)),
            'side_conv2': nn.Conv2d(128, 1, (1, 1)),
            'side_conv3': nn.Conv2d(256, 1, (1, 1)),
        })

        # Side outputs of each stage.
        self.side_outputs = None  # (B, 3, H, W)

        # Fused output.
        self.fused_output = None  # (B, 1, H, W)

        self.fusion_weights = None

        self._copy_weights_from_vgg16()

    def _copy_weights_from_vgg16(self):
        conv_layer_table = [
            ('conv1_1', 0),
            ('conv1_2', 2),
            ('conv2_1', 5),
            ('conv2_2', 7),
            ('conv3_1', 10),
            ('conv3_2', 12),
            ('conv3_3', 14),
        ]
        vgg_features = vgg16(pretrained=True).features

        for layer_name, layer_idx in conv_layer_table:
            vgg_layer = vgg_features[layer_idx]
            layer = self.vgg[layer_name]
            layer.weight.data = vgg_layer.weight.data
            layer.bias.data = vgg_layer.bias.data

    def forward(self, x):
        h = x

        h = F.relu(self.vgg['conv1_1'](h))
        h = F.relu(self.vgg['conv1_2'](h))
        side1 = torch.sigmoid(self.side['side_conv1'](h))
        h = self.vgg['pool1'](h)

        h = F.relu(self.vgg['conv2_1'](h))
        h = F.relu(self.vgg['conv2_2'](h))
        side2 = torch.sigmoid(self.side['side_conv2'](h))
        side2 = F.interpolate(side2, size=side1.size()[-2:],
                              mode='bilinear', align_corners=True)
        h = self.vgg['pool2'](h)

        h = F.relu(self.vgg['conv3_1'](h))
        h = F.relu(self.vgg['conv3_2'](h))
        h = F.relu(self.vgg['conv3_3'](h))
        side3 = torch.sigmoid(self.side['side_conv3'](h))
        side3 = F.interpolate(side3, size=side1.size()[-2:],
                              mode='bilinear', align_corners=True)

        # concatenate side outputs
        self.side_outputs = torch.cat(
            [side1, side2, side3], dim=1)  # (B, 3, H, W)

        if self.fusion_weights is None:
            weights = torch.tensor(self.kwargs.get('fusion_weights'))
            self.fusion_weights = weights.view(1, 3, 1, 1).to(x.device)

        self.fused_output = torch.sum(self.side_outputs * self.fusion_weights,
                                      dim=1, keepdim=True)  # (B, 1, H, W)

        return self.fused_output.squeeze(1)  # (B, H, W)


class CDWSTrainer(BaseTrainer):
    """Trainer for CDWS-MIL."""

    def __init__(self, model, **kwargs):
        """Initialize a trainer for CDWS-MIL.

        Kwargs:
            input_size: spatial size of input images
            side_ac_weights: weights of side area constrains loss
            fuse_ac_weight: weights of fused area constrains loss
            vgg_lr: learning rate of VGG
            side_lr: learning rate of side convolution layers
            gmp: generalized mean parameter
            weight_decay: weight decay for optimization
            epsilon: numerical stability term

        Returns:
            trainer: a new CDWSTrainer instance
        """

        config = CDWSConfig()
        kwargs = {**config.to_dict(), **kwargs}
        super().__init__(model, **kwargs)

    def get_default_dataset(self, root_dir, train=True, proportion=1.0):
        if train:
            return AreaConstraintDataset(root_dir, target_size=self.kwargs.get('input_size'),
                                         proportion=proportion)

        return SegmentationDataset(root_dir, train=False,
                                   target_size=self.kwargs.get('input_size'))

    def get_default_optimizer(self):
        optimizer = torch.optim.Adam([
            {'params': self.model.vgg.parameters()},
            {'params': self.model.side.parameters(), 'lr': self.kwargs.get('side_lr')}
        ], lr=self.kwargs.get('vgg_lr'), weight_decay=self.kwargs.get('weight_decay'))

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', patience=20, factor=0.5, min_lr=1e-5, verbose=True)

        return optimizer, None

    def preprocess(self, *data):
        data = [datum.to(self.device) for datum in data]
        if self.model.training:
            img, mask, area = data
            area = area[..., 0]
            target_class = (area > 0).float()
            return img, (mask, target_class, area)
        else:
            return data

    def compute_loss(self, pred, target, metrics=None):
        """Compute CWDS-MIL objective.

        Args:
            pred: model prediction
            target: a tuple containing following elements:
                1) pixel-level annotation of size (B, H, W)
                2) image-level labels of size (B,) (0 is background and 1 is foreground)
                3) relative sizes of foreground (between 0 and 1) of size (B,)

        Returns:
            loss: loss for side outputs and fused output
        """

        # Make sure that we cannot compute loss without a forward pass
        assert self.model.side_outputs is not None
        assert self.model.fused_output is not None

        _, target_class, target_area = target
        device = target_class.device
        epsilon = self.kwargs.get('epsilon')
        target_class = target_class.unsqueeze(-1)  # (B, 1)
        target_area = target_area.unsqueeze(-1)  # (B, 1)

        def mil_loss(output):
            output = output.clamp(min=epsilon, max=(1 - epsilon))
            image_pred = output.mean(
                dim=(2, 3)) ** (1 / self.kwargs.get('gmp'))  # (B, C)
            return target_class * -torch.log(image_pred) + \
                (1 - target_class) * -torch.log(1 - image_pred)

        def ac_loss(output):
            output = output.clamp(min=epsilon, max=(1 - epsilon))
            area_pred = output.mean(dim=(2, 3))  # (B, C)
            return target_class * (area_pred - target_area) ** 2

        side_mil_loss = mil_loss(self.model.side_outputs)  # (B, 3)
        side_ac_loss = ac_loss(self.model.side_outputs)  # (B, 3)
        side_ac_weights = torch.tensor(self.kwargs.get(
            'side_ac_weights')).to(device).unsqueeze(0)  # (1, 3)
        side_loss = side_mil_loss + side_ac_weights * side_ac_loss  # (B, 3)
        side_loss = torch.sum(side_loss, dim=1)  # (B,)

        fuse_mil_loss = mil_loss(self.model.fused_output)  # (B, 1)
        fuse_ac_loss = ac_loss(self.model.fused_output)  # (B, 1)
        fuse_loss = fuse_mil_loss + \
            self.kwargs.get('fuse_ac_weight') * fuse_ac_loss  # (B, 1)
        fuse_loss = fuse_loss.squeeze()  # (B,)

        metrics['side_loss'] = side_loss.mean().item()
        metrics['fuse_loss'] = fuse_loss.mean().item()

        loss = torch.mean(side_loss + fuse_loss)

        return loss

    def postprocess(self, pred, target=None):
        pred = pred.round().long()
        if target is not None:
            if self.model.training:
                target = target[0]
            return pred, target.argmax(dim=1)
        return pred

    def post_epoch_hook(self, epoch):
        if self.scheduler is not None:
            loss = np.mean(self.tracker.history['loss'])

            self.scheduler.step(loss)
