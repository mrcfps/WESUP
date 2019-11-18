import torch
import torch.nn as nn
from torchvision.models import vgg16

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

        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=100)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/2

        # conv2
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/4

        # conv3
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/8

        # conv4
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_3 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/16

        # conv5
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_3 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/32

        # fc6
        self.fc6 = nn.Conv2d(512, 4096, 7)
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout2d()

        # fc7
        self.fc7 = nn.Conv2d(4096, 4096, 1)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout2d()

        self.score_fr = nn.Conv2d(4096, 2, 1)
        self.upscore = nn.ConvTranspose2d(2, 2, 64, stride=32,
                                          bias=False)

        if checkpoint is not None:
            self.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.copy_params_from_vgg16()

        self.summary()

    def copy_params_from_vgg16(self):
        vgg = vgg16(pretrained=True)
        features = [
            self.conv1_1, self.relu1_1,
            self.conv1_2, self.relu1_2,
            self.pool1,
            self.conv2_1, self.relu2_1,
            self.conv2_2, self.relu2_2,
            self.pool2,
            self.conv3_1, self.relu3_1,
            self.conv3_2, self.relu3_2,
            self.conv3_3, self.relu3_3,
            self.pool3,
            self.conv4_1, self.relu4_1,
            self.conv4_2, self.relu4_2,
            self.conv4_3, self.relu4_3,
            self.pool4,
            self.conv5_1, self.relu5_1,
            self.conv5_2, self.relu5_2,
            self.conv5_3, self.relu5_3,
            self.pool5,
        ]
        for l1, l2 in zip(vgg.features, features):
            if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                assert l1.weight.size() == l2.weight.size()
                assert l1.bias.size() == l2.bias.size()
                l2.weight.data = l1.weight.data
                l2.bias.data = l1.bias.data
        for i, name in zip([0, 3], ['fc6', 'fc7']):
            l1 = vgg.classifier[i]
            l2 = getattr(self, name)
            l2.weight.data = l1.weight.data.view(l2.weight.size())
            l2.bias.data = l1.bias.data.view(l2.bias.size())

    def get_default_dataset(self, root_dir, train=True):
        if train:
            return PointSupervisionDataset(
                root_dir, target_size=config.input_size, include_obj_prior=True)

        return SegmentationDataset(root_dir, target_size=config.input_size, train=False)

    def get_default_optimizer(self, checkpoint=None):

        def biases():
            """Generator for fetching biases parameters."""
            for layer in self.children():
                if hasattr(layer, 'bias') and layer.bias is not None:
                    yield layer.bias

        def non_biases():
            """Generator for fetching non-biases parameters."""
            for layer in self.children():
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
        h = x
        h = self.relu1_1(self.conv1_1(h))
        h = self.relu1_2(self.conv1_2(h))
        h = self.pool1(h)

        h = self.relu2_1(self.conv2_1(h))
        h = self.relu2_2(self.conv2_2(h))
        h = self.pool2(h)

        h = self.relu3_1(self.conv3_1(h))
        h = self.relu3_2(self.conv3_2(h))
        h = self.relu3_3(self.conv3_3(h))
        h = self.pool3(h)

        h = self.relu4_1(self.conv4_1(h))
        h = self.relu4_2(self.conv4_2(h))
        h = self.relu4_3(self.conv4_3(h))
        h = self.pool4(h)

        h = self.relu5_1(self.conv5_1(h))
        h = self.relu5_2(self.conv5_2(h))
        h = self.relu5_3(self.conv5_3(h))
        h = self.pool5(h)

        h = self.relu6(self.fc6(h))
        h = self.drop6(h)

        h = self.relu7(self.fc7(h))
        h = self.drop7(h)

        h = self.score_fr(h)

        h = self.upscore(h)
        h = h[:, :, 19:19 + x.size()[2], 19:19 + x.size()[3]].contiguous()

        return h[:, 1, ...]

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
