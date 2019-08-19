import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.data import SegmentationDataset
from .base import BaseConfig, BaseModel


class MILDNetConfig(BaseConfig):
    """Configuration for CWDS-MIL model."""

    # Input spatial size.
    input_size = (464, 464)

    # Number of epochs for decaying auxillary loss.
    aux_decay_period = 8

    # Initial learning rate.
    initial_lr = 1e-4

    # L2 weight decay.
    weight_decay = 1e-5

    # numerical stability term
    epsilon = 1e-7


class MILUnit(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1_1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.conv2_1 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        self.conv1_2 = nn.Conv2d(3, in_ch, 3, 1, 1)
        self.conv2_2 = nn.Conv2d(2 * in_ch, out_ch, 3, 1, 1)

    def forward(self, feature_maps, d_img):
        d_img = F.relu(self.conv1_2(d_img))
        branch_1 = self.conv2_1(F.relu(self.conv1_1(feature_maps)))
        branch_2 = F.relu(self.conv2_2(torch.cat([feature_maps, d_img], dim=1)))
        return F.relu(branch_1 + branch_2)


class ResidualUnit(nn.Module):
    def __init__(self, in_ch, out_ch, dilation_rate=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, 1,
                               padding=dilation_rate, dilation=dilation_rate)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1,
                               padding=dilation_rate, dilation=dilation_rate)

    def forward(self, x):
        return F.relu(x + self.conv2(F.relu(self.conv1(x))))


class ASPP(nn.Module):
    """
    Atrous spatial pyramid pooling (ASPP) module.
    """

    def __init__(self, in_ch, out_ch, rates=(6, 12, 18, 24)):
        super().__init__()
        self.in_channels = in_ch
        self.out_channels = out_ch
        self.rates = rates

        for idx, rate in enumerate(rates):
            setattr(self, f'branch_{idx}', self._build_branch(rates[idx]))

    def _build_branch(self, rate):
        branch = nn.Sequential(
            nn.Conv2d(self.in_channels, 1024, 3, stride=1,
                      padding=rate, dilation=rate),
            nn.ReLU(inplace=False),
            nn.Dropout2d(p=0.5),
            nn.Conv2d(1024, 1024, 1, stride=1),
            nn.ReLU(inplace=False),
            nn.Dropout2d(p=0.5),
            nn.Conv2d(1024, self.out_channels, 1, stride=1),
        )

        # special initialization for the last conv layer
        nn.init.normal_(branch[-1].weight, mean=0, std=0.01)
        nn.init.constant_(branch[-1].bias, 0)

        return branch

    def forward(self, x):
        return sum(getattr(self, f'branch_{idx}')(x) for idx in range(len(self.rates)))


class MILDNet(BaseModel):
    """
    MILD-Net: Minimal Information Loss Dilated Network for Gland Instance
    Segmentation in Colon Histology Images.

    See https://arxiv.org/pdf/1806.01963.pdf.
    """

    def __init__(self, checkpoint=None):
        super().__init__()

        self.config = MILDNetConfig()
        self.conv1_1 = nn.Conv2d(3, 64, 3, 1, 1)
        self.conv1_2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.mil1 = MILUnit(64, 128)
        self.res1 = ResidualUnit(128, 128)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.mil2 = MILUnit(128, 256)
        self.res2 = ResidualUnit(256, 256)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.mil3 = MILUnit(256, 512)
        self.res3 = ResidualUnit(512, 512)

        self.dilated_res1 = ResidualUnit(512, 512, dilation_rate=2)
        self.dilated_res2 = ResidualUnit(512, 512, dilation_rate=2)
        self.dilated_res3 = ResidualUnit(512, 512, dilation_rate=4)
        self.dilated_res4 = ResidualUnit(512, 512, dilation_rate=4)

        self.aux_obj_conv = nn.Conv2d(512, 2, 1, 1, 0)
        self.aux_cont_conv = nn.Conv2d(512, 2, 1, 1, 0)

        self.aspp = ASPP(512, 640, rates=(6, 12, 18))

        # upsampling
        self.conv2_1 = nn.Conv2d(640 + 256, 256, 3, 1, 1)
        self.conv2_2 = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv3_1 = nn.Conv2d(256 + 128, 128, 3, 1, 1)
        self.conv3_2 = nn.Conv2d(128, 128, 3, 1, 1)

        # object branch
        self.obj_conv1 = nn.Conv2d(128 + 64, 64, 3, 1, 1)
        self.obj_conv2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.obj_dropout = nn.Dropout2d(p=0.5)
        self.obj_conv3 = nn.Conv2d(64, 2, 1, 1, 0)

        # contour branch
        self.cont_conv1 = nn.Conv2d(128 + 64, 64, 3, 1, 1)
        self.cont_conv2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.cont_dropout = nn.Dropout2d(p=0.5)
        self.cont_conv3 = nn.Conv2d(64, 2, 1, 1, 0)

        self.epoch = 0

        if checkpoint is not None:
            self.load_state_dict(checkpoint['model_state_dict'])
            self.epoch = checkpoint['epoch']

    def get_default_dataset(self, root_dir, train=True, proportion=1.0):
        if train:
            return SegmentationDataset(root_dir, target_size=self.config.input_size,
                                       contour=True, proportion=proportion)

        return SegmentationDataset(root_dir, target_size=self.config.input_size, train=False)

    def get_default_optimizer(self, checkpoint=None):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.initial_lr,
                                     weight_decay=self.config.weight_decay)

        if checkpoint is not None:
            # load previous optimizer states
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        return optimizer, None

    def preprocess(self, *data, device='cpu'):
        data = [datum.to(device) for datum in data]
        if self.training:
            img, mask, cont = data
            return img, (mask.float(), cont.float())
        else:
            return data

    def forward(self, x):
        orig_img = x
        h = x

        h = F.relu(self.conv1_1(h))
        h = F.relu(self.conv1_2(h))
        low1 = h
        h = self.pool1(h)

        h = self.res1(self.mil1(h, F.interpolate(orig_img, scale_factor=0.5)))
        low2 = h
        h = self.pool2(h)

        h = self.res2(self.mil2(h, F.interpolate(orig_img, scale_factor=0.25)))
        low3 = h
        h = self.pool3(h)

        h = self.res3(self.mil3(h, F.interpolate(orig_img, scale_factor=0.125)))
        h = self.dilated_res1(h)
        h = self.dilated_res2(h)
        aux_obj = F.softmax(self.aux_obj_conv(h), dim=1)
        aux_cont = F.softmax(self.aux_cont_conv(h), dim=1)

        h = self.dilated_res3(h)
        h = self.dilated_res4(h)
        h = self.aspp(h)

        h = torch.cat([F.interpolate(h, scale_factor=2), low3], dim=1)
        h = F.relu(self.conv2_1(h))
        h = F.relu(self.conv2_2(h))

        h = torch.cat([F.interpolate(h, scale_factor=2), low2], dim=1)
        h = F.relu(self.conv3_1(h))
        h = F.relu(self.conv3_2(h))

        h = torch.cat([F.interpolate(h, scale_factor=2), low1], dim=1)

        obj = F.relu(self.obj_conv1(h))
        obj = F.relu(self.obj_conv2(obj))
        obj = self.obj_conv3(self.obj_dropout(obj))
        obj = F.softmax(obj, dim=1)

        cont = F.relu(self.cont_conv1(h))
        cont = F.relu(self.cont_conv2(cont))
        cont = self.cont_conv3(self.cont_dropout(cont))
        cont = F.softmax(cont, dim=1)

        return obj, cont, aux_obj, aux_cont

    def compute_loss(self, pred, target, metrics=None):
        """Compute CWDS-MIL objective.

        Args:
            pred: model prediction with following elements:
                1) final object prediction of size (B, C, H, W)
                2) final contour prediction of size (B, C, H, W)
                3) auxillary object prediction of size (B, C, H/8, W/8)
                4) auxillary contour prediction of size (B, C, H/8, W/8)
            target: a tuple containing following elements:
                1) object ground truth of size (B, C, H, W)
                2) contour ground truth of size (B, C, H, W)

        Returns:
            loss: loss for side outputs and fused output
        """

        obj, cont, aux_obj, aux_cont = pred
        obj_gt, cont_gt = target
        aux_obj_gt = F.interpolate(obj_gt, scale_factor=1/8)
        aux_cont_gt = F.interpolate(cont_gt, scale_factor=1/8)

        def cross_entropy(p, g):
            return torch.mean(g * -torch.log(torch.clamp_min(p, min=self.config.epsilon)))

        obj_loss = cross_entropy(obj, obj_gt)
        cont_loss = cross_entropy(cont, cont_gt)
        aux_obj_loss = cross_entropy(aux_obj, aux_obj_gt)
        aux_cont_loss = cross_entropy(aux_cont, aux_cont_gt)

        decay = 0.1 ** (self.epoch // self.config.aux_decay_period)
        metrics['aux_decay'] = decay

        return obj_loss + cont_loss + decay * (aux_obj_loss + aux_cont_loss)

    def postprocess(self, pred, target=None):
        obj, cont, _, _ = pred
        obj = obj.round().long()
        cont = cont.round().long()

        # fusion of predicted objects and contours
        pred = obj & (1 - cont)

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
