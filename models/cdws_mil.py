import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16

from utils.data import SegmentationDataset
from utils.data import AreaConstraintDataset
from .base import BaseModel


class CDWSConfig:
    """Configuration for CWDS-MIL model."""

    # Input spatial size.
    input_size = (280, 400)

    # Fixed fusion weights
    fusion_weights = torch.tensor([0.2, 0.35, 0.45]).view(1, 3, 1, 1)

    # Weights of area constrains loss
    side_ac_weights = torch.tensor([[2.5, 5, 10]])
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


config = CDWSConfig()


class CDWS(BaseModel):
    """
    Constrained deep weak supervision network for histopathology image segmentation.
    See https://arxiv.org/pdf/1701.00794.pdf.
    """

    def __init__(self, checkpoint=None):
        super().__init__()

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

        if checkpoint is not None:
            self.load_state_dict(checkpoint['model_state_dict'])
        else:
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

    def get_default_config(self):
        return {
            k: v for k, v in CDWSConfig.__dict__.items()
            if not k.startswith('__')
        }

    def get_default_dataset(self, root_dir, train=True):
        if train:
            return AreaConstraintDataset(root_dir, target_size=config.input_size)

        return SegmentationDataset(root_dir, target_size=config.input_size, train=False)

    def get_default_optimizer(self, checkpoint=None):
        optimizer = torch.optim.Adam([
            {'params': self.vgg.parameters()},
            {'params': self.side.parameters(), 'lr': config.side_lr}
        ], lr=config.vgg_lr, weight_decay=config.weight_decay)

        if checkpoint is not None:
            # load previous optimizer states
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        return optimizer, None

    def preprocess(self, *data):
        if self.training:
            img, mask, area = data
            target_class = (area > 0).float()
            return img, (mask, target_class, area)
        else:
            return data

    def forward(self, x):
        h = x

        h = F.relu(self.vgg['conv1_1'](h))
        h = F.relu(self.vgg['conv1_2'](h))
        side1 = torch.sigmoid(self.side['side_conv1'](h))
        h = self.vgg['pool1'](h)

        h = F.relu(self.vgg['conv2_1'](h))
        h = F.relu(self.vgg['conv2_2'](h))
        side2 = torch.sigmoid(self.side['side_conv2'](h))
        side2 = F.interpolate(side2, size=side1.size()[-2:], mode='bilinear')
        h = self.vgg['pool2'](h)

        h = F.relu(self.vgg['conv3_1'](h))
        h = F.relu(self.vgg['conv3_2'](h))
        h = F.relu(self.vgg['conv3_3'](h))
        side3 = torch.sigmoid(self.side['side_conv3'](h))
        side3 = F.interpolate(side3, size=side1.size()[-2:], mode='bilinear')

        # concatenate side outputs
        self.side_outputs = torch.cat([side1, side2, side3], dim=1)  # (B, 3, H, W)

        fusion_weights = config.fusion_weights.to(x.device).detach()
        self.fused_output = torch.sum(self.side_outputs * fusion_weights,
                                      dim=1, keepdim=True)  # (B, 1, H, W)

        return self.fused_output.squeeze(1)  # (B, H, W)

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
        assert self.side_outputs is not None
        assert self.fused_output is not None

        _, target_class, target_area = target
        device = target_class.device
        target_class = target_class.unsqueeze(-1)  # (B, 1)
        target_area = target_area.unsqueeze(-1)  # (B, 1)

        def mil_loss(output):
            output = output.clamp(min=config.epsilon, max=(1 - config.epsilon))
            image_pred = output.mean(dim=(2, 3)) ** (1 / config.gmp)  # (B, C)
            return target_class * -torch.log(image_pred) + \
                (1 - target_class) * -torch.log(1 - image_pred)

        def ac_loss(output):
            output = output.clamp(min=config.epsilon, max=(1 - config.epsilon))
            area_pred = output.mean(dim=(2, 3))  # (B, C)
            return target_class * (area_pred - target_area) ** 2

        side_mil_loss = mil_loss(self.side_outputs)  # (B, 3)
        side_ac_loss = ac_loss(self.side_outputs)  # (B, 3)
        side_ac_weights = config.side_ac_weights.to(device).detach()  # (1, 3)
        side_loss = side_mil_loss + side_ac_weights * side_ac_loss  # (B, 3)
        side_loss = torch.sum(side_loss, dim=1)  # (B,)

        fuse_mil_loss = mil_loss(self.fused_output)  # (B, 1)
        fuse_ac_loss = ac_loss(self.fused_output)  # (B, 1)
        fuse_loss = fuse_mil_loss + config.fuse_ac_weight * fuse_ac_loss  # (B, 1)
        fuse_loss = fuse_loss.squeeze()  # (B,)

        return torch.mean(side_loss + fuse_loss)

    def postprocess(self, pred, target=None):
        pred = pred.round().long()
        if target is not None:
            if self.training:
                target = target[0]
            return pred, target.argmax(dim=1)
        return pred

    def save_checkpoint(self, ckpt_path, **kwargs):
        """Save model checkpoint."""

        torch.save({
            'model_state_dict': self.state_dict(),
            **kwargs,
        }, ckpt_path)
        print(f'Checkpoint saved to {ckpt_path}.')
