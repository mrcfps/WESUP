import torch
import torch.nn as nn
import numpy as np
from torchvision.models import vgg16

from utils.data import SegmentationDataset
from .base import BaseConfig, BaseTrainer


class FCNConfig(BaseConfig):
    n_classes = 2
    target_size = (320, 320)

    # Optimization parameters.
    momentum = 0.9
    weight_decay = 0.0005


class FCN32s(nn.Module):

    def __init__(self, n_classes=2):
        super().__init__()
        self.layer1 = self.conv_sequential_double(3,64,3,1,2,2)
        self.layer2 = self.conv_sequential_double(64, 128, 3, 1, 2, 2)
        self.layer3 = self.conv_sequential_triple(128, 256, 3, 1, 2, 2)
        self.layer4 = self.conv_sequential_triple(256, 512, 3, 1, 2, 2)
        self.layer5 = self.conv_sequential_triple(512, 512, 3, 1, 2, 2)
        self.transpose_layer2 = self.transpose_conv_sequential(3,3,4,2,1)
        self.transpose_layer8 = self.transpose_conv_sequential(3,n_classes,16,8,4)
        self.ravel_layer32 = nn.Sequential(
            nn.Conv2d(512,3,1),
            nn.ReLU(True)
        )
        self.ravel_layer16 = nn.Sequential(
            nn.Conv2d(512,3,1),
            nn.ReLU(True)
        )
        self.ravel_layer8 = nn.Sequential(
            nn.Conv2d(256,3,1),
            nn.ReLU(True)
        )
    def forward(self,x):
        ret = self.layer1(x)
        ret = self.layer2(ret)
        ret = self.layer3(ret)
        x8 = ret
        ret = self.layer4(ret)
        x16 = ret
        ret = self.layer5(ret)
        x32 = ret
        x32 = self.ravel_layer32(x32)
        x16 = self.ravel_layer16(x16)
        x8 = self.ravel_layer8(x8)
        x32 = self.transpose_layer2(x32)
        x16 =x16+x32
        x16 = self.transpose_layer2(x16)
        x8 =x8+x16
        result = self.transpose_layer8(x8)
        return nn.functional.softmax(result,dim=1)

    def conv_sequential_double(self,in_size,out_size,kfilter,padding,kernel_size,stride):
        return nn.Sequential(
            nn.Conv2d(in_size,out_size,kfilter,padding=padding),
            nn.BatchNorm2d(out_size),
            nn.ReLU(True),
            nn.Conv2d(out_size,out_size,kfilter,padding=padding),
            nn.BatchNorm2d(out_size),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size,stride)
        )
    
    def conv_sequential_triple(self,in_size,out_size,kfilter,padding,kernel_size,stride):
        return nn.Sequential(
            nn.Conv2d(in_size,out_size,kfilter,padding=padding),
            nn.BatchNorm2d(out_size),
            nn.ReLU(True),
            nn.Conv2d(out_size,out_size,kfilter,padding=padding),
            nn.BatchNorm2d(out_size),
            nn.ReLU(True),
            nn.Conv2d(out_size,out_size,kfilter,padding=padding),
            nn.BatchNorm2d(out_size),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size,stride)
        )

    def transpose_conv_sequential(self,in_size,out_size,kfilter,stride,padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_size,out_size,kfilter,stride,padding,bias=False),
            nn.BatchNorm2d(out_size)
        )


class FCNTrainer(BaseTrainer):
    """Trainer for FCN."""

    def __init__(self, model, **kwargs):
        """Initialize a MILDNetTrainer.
        Kwargs:
            input_size: input spatial size
            contour_threshold: threshold for predicting contours
            aux_decay_period: number of epochs for decaying auxillary loss
            initial_lr: initial learning rate
            weight_decay: weight decay for optimizer
            epsilon: numerical stability term
        Returns:
            trainer: a new MILDNetTrainer instance
        """

        super().__init__(model, **kwargs)
        self.kwargs = kwargs

    def get_default_dataset(self, root_dir, train=True, proportion=1.0):
        if train:
            return SegmentationDataset(root_dir, target_size=self.kwargs.get('target_size'))

        return SegmentationDataset(root_dir, train=False,
                                   target_size=self.kwargs.get('target_size'))

    def get_default_optimizer(self):
        optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=1e-3,
            momentum=0.9,
            weight_decay=self.kwargs.get('weight_decay'),
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', patience=20, factor=0.5, min_lr=1e-5, verbose=True)

        return optimizer, scheduler

    def preprocess(self, *data):
        return [datum.to(self.device) for datum in data]

    def compute_loss(self, pred, target, metrics=None):
        """Compute FCN objective.
        Args:
            pred: model prediction of size (B, C, H, W)
            target: ground truth mask of size (B, C, H, W)
        Returns:
            loss: pixel-wise classification loss
        """

        pred = torch.clamp(pred, min=1e-7, max=1-1e-7)
        return torch.mean(target.float() * -torch.log(pred))

    def postprocess(self, pred, target=None):
        pred = pred.round()[:, 1, ...].long()

        if target is not None:
            return pred, target.argmax(dim=1)

        return pred
    
    def post_epoch_hook(self, epoch):
        if self.scheduler is not None:
            loss = np.mean(self.tracker.history['loss'])

            self.scheduler.step(loss)
