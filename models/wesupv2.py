import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.data import SegmentationDataset
from utils.data import WESUPV2Dataset
from .base import BaseConfig, BaseTrainer


class WESUPV2Config(BaseConfig):
    """Configuration for WESUPV2 model."""

    # Input spatial size.
    input_size = (280, 400)

    # Output dimension of concatenated feature maps.
    D = 8

    # Feature map size.
    fm_size = (35, 50)

    beta = 2

    transition_iterations = 4

    # Threshold for predicting contours.
    contour_threshold = 0.3

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


class WESUPV2(nn.Module):
    """
    MILD-Net: Minimal Information Loss Dilated Network for Gland Instance
    Segmentation in Colon Histology Images.

    See https://arxiv.org/pdf/1806.01963.pdf.
    """

    def __init__(self, D=32, fm_size=(280, 400), **kwargs):
        """Initialize WESUPV2.

        Args:
            D: # channels of fused output feature map
        """

        super().__init__()
        self.fm_size = fm_size

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

        # cross convolutions
        self.cross_conv1 = nn.Conv2d(3968, 512, 3, 1, 1)
        self.cross_conv2 = nn.Conv2d(512, D, 3, 1, 1)

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

        self.epoch = 0

    def forward(self, x):
        orig_img = x
        h = x

        h = F.relu(self.conv1_1(h))
        fmaps = F.interpolate(h, self.fm_size)
        h = F.relu(self.conv1_2(h))
        fmaps = torch.cat([fmaps, F.interpolate(h, self.fm_size)], dim=1)
        low1 = h
        h = self.pool1(h)

        h = self.mil1(h, F.interpolate(orig_img, scale_factor=0.5))
        fmaps = torch.cat([fmaps, F.interpolate(h, self.fm_size)], dim=1)
        h = self.res1(h)
        fmaps = torch.cat([fmaps, F.interpolate(h, self.fm_size)], dim=1)
        low2 = h
        h = self.pool2(h)

        h = self.mil2(h, F.interpolate(orig_img, scale_factor=0.25))
        fmaps = torch.cat([fmaps, F.interpolate(h, self.fm_size)], dim=1)
        h = self.res2(h)
        fmaps = torch.cat([fmaps, F.interpolate(h, self.fm_size)], dim=1)
        low3 = h
        h = self.pool3(h)

        h = self.mil3(h, F.interpolate(orig_img, scale_factor=0.125))
        fmaps = torch.cat([fmaps, F.interpolate(h, self.fm_size)], dim=1)
        h = self.res3(h)
        fmaps = torch.cat([fmaps, F.interpolate(h, self.fm_size)], dim=1)
        h = self.dilated_res1(h)
        fmaps = torch.cat([fmaps, F.interpolate(h, self.fm_size)], dim=1)
        h = self.dilated_res2(h)
        fmaps = torch.cat([fmaps, F.interpolate(h, self.fm_size)], dim=1)
        h = self.dilated_res3(h)
        fmaps = torch.cat([fmaps, F.interpolate(h, self.fm_size)], dim=1)
        h = self.dilated_res4(h)
        fmaps = torch.cat([fmaps, F.interpolate(h, self.fm_size)], dim=1)
        h = self.aspp(h)

        h = torch.cat([F.interpolate(h, scale_factor=2), low3], dim=1)
        h = F.relu(self.conv2_1(h))
        h = F.relu(self.conv2_2(h))

        h = torch.cat([F.interpolate(h, scale_factor=2), low2], dim=1)
        h = F.relu(self.conv3_1(h))
        h = F.relu(self.conv3_2(h))

        h = torch.cat([F.interpolate(h, scale_factor=2), low1], dim=1)

        h = F.relu(self.obj_conv1(h))
        h = F.relu(self.obj_conv2(h))
        h = self.obj_conv3(self.obj_dropout(h))
        h = F.softmax(h, dim=1)

        fmaps = F.relu(self.cross_conv1(fmaps))
        fmaps = F.relu(self.cross_conv2(fmaps))

        return h, fmaps


class WESUPV2Trainer(BaseTrainer):
    """Trainer for WESUPV2."""

    def __init__(self, model, **kwargs):
        """Initialize a MILDNetTrainer.

        Kwargs:
            input_size: input spatial size
            transition_iteration: number of iterations for random walk transition
            contour_threshold: threshold for predicting contours
            aux_decay_period: number of epochs for decaying auxillary loss
            initial_lr: initial learning rate
            weight_decay: weight decay for optimizer
            epsilon: numerical stability term

        Returns:
            trainer: a new MILDNetTrainer instance
        """

        kwargs = {**WESUPV2Config().to_dict(), **kwargs}
        super().__init__(model, **kwargs)

    def get_default_dataset(self, root_dir, train=True, proportion=1.0):
        if train:
            return WESUPV2Dataset(root_dir, target_size=self.kwargs.get('input_size'),
                                  proportion=proportion)

        return SegmentationDataset(root_dir, train=False,
                                   target_size=self.kwargs.get('input_size'))

    def get_default_optimizer(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.kwargs.get('initial_lr'),
                                     weight_decay=self.kwargs.get('weight_decay'))

        return optimizer, None

    def preprocess(self, *data):
        data = [datum.to(self.device) for datum in data]
        if self.model.training:
            img, mask, coords = data
            return img, (mask.float(), coords)
        else:
            return data

    def compute_loss(self, pred, target, metrics=None):
        """Compute WESUPV2 objective.

        Args:
            pred: model prediction with following elements:
                1) segmentation prediction (B, C, H_o, W_o)
                2) feature maps of size (B, D, H_f, W_f)
            target: a tuple containing following elements:
                1) partially labeled mask of size (B, C, H_o, W_o)
                2) coordinates array of size (B, 2, H_o x W_o)

        Returns:
            loss: loss for side outputs and fused output
        """

        h, fmaps = pred
        mask, coords = target
        batch, D, height_f, width_f = fmaps.size()
        _, C, height_o, width_o = mask.size()

        # compute feature-wise affinity matrix
        fmaps = fmaps.view(batch, D, 1, -1)
        fmaps_p = fmaps.view(batch, D, -1, 1)
        feature_aff = torch.exp(-torch.einsum('ijkl,ijkl->ikl',
                                              fmaps - fmaps_p, fmaps - fmaps_p) / 2)

        # compute spatial adjacency matrix
        coords = F.interpolate(coords, fmaps.size()[2:], mode='bilinear',
                               align_corners=True)
        coords = coords.view(batch, 2, 1, -1)
        coords_p = coords.view(batch, 2, -1, 1)
        coords_aff = torch.exp(-torch.einsum('ijkl,ijkl->ikl',
                                             coords - coords_p, coords - coords_p) / 2)

        # TODO: check if matrix aff is all ones on diagonal.
        aff = feature_aff * coords_aff
        aff = torch.pow(aff, self.kwargs.get('beta'))  # (B, H x W, H x W)
        transition = aff / aff.sum(dim=1, keepdim=True)  # (B, H x W, H x W)
        mask = F.interpolate(mask, (height_f, width_f), mode='nearest')
        mask = mask.view(batch, -1, height_f * width_f).float()  # (B, C, H x W)

        for _ in range(self.kwargs.get('transition_iterations')):
            mask = torch.bmm(mask, transition)  # (B, C, H x W)

        mask = mask.view(batch, C, height_f, width_f)
        mask = F.interpolate(mask, (height_o, width_o),
                             mode='bilinear', align_corners=True)

        h = torch.clamp(h, self.kwargs.get('epsilon'), 1 - self.kwargs.get('epsilon'))

        return torch.sum(mask * -torch.log(h)) / mask.nonzero().size(0)

    def postprocess(self, pred, target=None):
        pred = pred[0].round().long()

        if self.model.training:
            return pred, None
        return pred, target.argmax(dim=1)
