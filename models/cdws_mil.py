import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16

from .base import BaseModel


class CDWS(BaseModel):
    """
    Constrained deep weak supervision network for histopathology image segmentation.
    See https://arxiv.org/pdf/1701.00794.pdf.
    """

    # Fixed fusion weights
    fusion_weights = torch.tensor([0.2, 0.35, 0.45]).view(1, 3, 1, 1)

    # Weights of area constrains loss
    side_ac_weights = torch.tensor([[2.5, 5, 10]])
    fuse_ac_weight = 10

    # Generalized mean parameter
    gmp = 4

    def __init__(self, checkpoint=None):
        super().__init__()

        # First stage
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, stride=2)

        # Second stage
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, stride=2)

        # Third stage
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)

        # Side output conv layers.
        self.side_conv1 = nn.Conv2d(64, 1, (1, 1))
        self.side_conv2 = nn.Conv2d(128, 1, (1, 1))
        self.side_conv3 = nn.Conv2d(256, 1, (1, 1))

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
            (self.conv1_1, 0),
            (self.conv1_2, 2),
            (self.conv2_1, 5),
            (self.conv2_2, 7),
            (self.conv3_1, 10),
            (self.conv3_2, 12),
            (self.conv3_3, 14),
        ]
        vgg_features = vgg16(pretrained=True).features

        for layer, layer_idx in conv_layer_table:
            vgg_layer = vgg_features[layer_idx]
            layer.weight.data = vgg_layer.weight.data
            layer.bias.data = vgg_layer.bias.data

    def preprocess(self, *data):
        if self.training:
            img, mask, area = data
            target_class = (area > 0).long()
            return img, (mask, target_class, area)
        else:
            return data

    def forward(self, x):
        h = x

        h = F.relu(self.conv1_1(h))
        h = F.relu(self.conv1_2(h))
        side1 = F.sigmoid(self.side_conv1(h))
        h = self.pool1(h)

        h = F.relu(self.conv2_1(h))
        h = F.relu(self.conv2_2(h))
        side2 = torch.sigmoid(self.side_conv2(h))
        side2 = F.interpolate(side2, size=side1.size()[-2:], mode='bilinear')
        h = self.pool2(h)

        h = F.relu(self.conv3_1(h))
        h = F.relu(self.conv3_2(h))
        h = F.relu(self.conv3_3(h))
        side3 = torch.sigmoid(self.side_conv3(h))
        side3 = F.interpolate(side3, size=side1.size()[-2:], mode='bilinear')

        # concatenate side outputs
        self.side_outputs = torch.cat([side1, side2, side3], dim=1)

        fusion_weights = self.fusion_weights.to(x.device).detach()
        self.fused_output = torch.sum(self.side_outputs * fusion_weights,
                                      dim=1, keepdim=True)

        return self.fused_output.squeeze(0)  # (1, H, W)

    def compute_loss(self, pred, target, metrics=None):
        """Compute CWDS-MIL objective.

        Args:
            pred: model prediction
            target: a tuple containing image-level labels (should be a tensor of size (B,),
                with 0 represents background and 1 represents foreground) and relative sizes
                (between 0 and 1) of foreground (should be a tensor of size (B,))

        Returns:
            loss: loss for side outputs and fused output
        """

        # Make sure that we cannot compute loss without a forward pass
        assert self.side_outputs is not None
        assert self.fused_output is not None

        _, target_class, target_area = target
        device = target_class.device
        target_class = target_class.unsqueeze(-1).float()  # (B, 1)
        target_area = target_area.unsqueeze(-1)  # (B, 1)

        def mil_loss(output):
            image_pred = output.mean(dim=(2, 3)) ** (1 / self.gmp)  # (B, C)
            return target_class * torch.log(image_pred) + \
                (1 - target_class) * torch.log(1 - image_pred)

        def ac_loss(output):
            area_pred = output.mean(dim=(2, 3))  # (B, C)
            return target_class * (area_pred - target_area) ** 2

        side_mil_loss = mil_loss(self.side_outputs)  # (B, 3)
        side_ac_loss = ac_loss(self.side_outputs)  # (B, 3)
        self.side_ac_weights = self.side_ac_weights.to(device).detach()  # (1, 3)
        side_loss = side_mil_loss + self.side_ac_weights * side_ac_loss  # (B, 3)
        side_loss = torch.sum(side_loss, dim=1)  # (B,)

        fuse_mil_loss = mil_loss(self.fused_output)  # (B, 1)
        fuse_ac_loss = ac_loss(self.fused_output)  # (B, 1)
        fuse_loss = fuse_mil_loss + self.fuse_ac_weight * fuse_ac_loss  # (B, 1)
        fuse_loss = fuse_loss.squeeze()  # (B,)

        return torch.mean(side_loss + fuse_loss)

    def _pre_evaluate_hook(self, pred, target):
        return pred.round().long(), target[0]

    def save_checkpoint(self, ckpt_path, **kwargs):
        """Save model checkpoint."""

        torch.save({
            'model_state_dict': self.state_dict(),
            **kwargs,
        }, ckpt_path)
        print(f'Checkpoint saved to {ckpt_path}.')
