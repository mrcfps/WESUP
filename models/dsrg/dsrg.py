import torch
import torch.nn.functional as F
import numpy as np
import torchvision.transforms.functional as TF
from skimage.measure import label

from utils.data import SegmentationDataset
from utils.data import PointSupervisionDataset
from .deeplabv2 import DeepLabV2
from .densecrf import dense_crf
from ..base import BaseConfig, BaseModel


class DSRGConfig(BaseConfig):
    """Configuration for DSRG model."""

    # Input spatial size.
    input_size = (288, 400)

    # number of target classes.
    n_classes = 2

    # Similarity criteria for region growing
    # (b: background, f: foreground)
    theta_b = 0.65
    theta_f = 0.65

    # learning rate
    initial_lr = 5e-4

    # weight decay for optimization
    weight_decay = 5e-4

    # numerical stability term
    epsilon = 1e-7


class DSRG(BaseModel):
    """
    Weakly-Supervised Semantic Segmentation Network with Deep Seeded Region Growing (CVPR2018).
    """

    def __init__(self, checkpoint=None):
        super().__init__()

        self.config = DSRGConfig()
        self.seg_net = DeepLabV2(self.config.n_classes)

        if checkpoint is not None:
            self.load_state_dict(checkpoint['model_state_dict'])

    def get_default_dataset(self, root_dir, train=True, proportion=1.0):
        if train:
            return PointSupervisionDataset(root_dir, proportion=proportion, radius=10,
                                           target_size=self.config.input_size)

        return SegmentationDataset(root_dir, target_size=self.config.input_size, train=False)

    def get_default_optimizer(self, checkpoint=None):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.config.initial_lr,
                                    weight_decay=self.config.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 500)

        if checkpoint is not None:
            # load previous optimizer states
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        return optimizer, scheduler

    def preprocess(self, *data, device='cpu'):
        data = [datum.to(device) for datum in data]
        if self.training:
            img, pixel_mask, point_mask = data

            # img is also attached to targets list for computing CRF
            return img, (img, pixel_mask, point_mask)

        img, mask = data
        return img, (img, mask)

    def forward(self, x):
        return self.seg_net(x)

    def compute_loss(self, pred, target, metrics=None):
        """Compute DSRG objective.

        Args:
            pred: segmentation network prediction (B, C, H, W)
            target: a tuple containing following elements:
                1) input image of size (B, 3, H, W)
                2) pixel-level annotation of size (B, C, H, W)
                3) point annotation of size (B, C, H, W)

        Returns:
            loss: seeding loss + boundary loss
        """

        pred = torch.clamp_min(pred, min=self.config.epsilon)
        img, _, point_mask = target

        t_pred = pred.detach().cpu().numpy()
        t_pred[:, 0, ...] = t_pred[:, 0, ...] > self.config.theta_b
        t_pred[:, 1:, ...] = t_pred[:, 1:, ...] > self.config.theta_f
        grown = np.zeros_like(t_pred)  # (B, C, H, W)

        # how much we would like to multiply coordinates of each point
        multiplier = (grown.shape[2] / point_mask.size(2), grown.shape[3] / point_mask.size(3))

        # perform region growing for each class
        for class_idx in range(pred.size(1)):
            cls_point_mask = point_mask[:, class_idx, ...].cpu().numpy()
            bs, hs, ws = np.where(cls_point_mask > 0)
            regions = label(t_pred[:, class_idx, ...])

            for b, h, w in zip(bs, hs, ws):
                h = int(h * multiplier[0])
                w = int(w * multiplier[1])

                if regions[b, h, w] > 0:
                    region_mask = regions == regions[b, h, w]
                    grown[:, class_idx, ...][region_mask] = 1
                else:
                    grown[b, class_idx, h, w] = 1

            metrics[f'grown_{class_idx}'] = grown[:, class_idx, ...].mean()

        grown = torch.Tensor(grown).to(pred.device)
        seeding_loss = torch.mean(grown * -torch.log(pred))

        h, w = pred.size(2), pred.size(3)
        img = [np.array(TF.to_pil_image(i.cpu()).resize((w, h))) for i in img]

        crf_outputs = []
        for probs, image in zip(pred, img):
            probs = probs.detach().cpu().numpy()
            probs = np.transpose(probs, (1, 2, 0))
            crf_outputs.append(dense_crf(probs, image))

        crf_outputs = torch.Tensor(crf_outputs).to(pred.device)
        boundary_loss = torch.mean(crf_outputs * torch.log(crf_outputs / pred))

        return seeding_loss + boundary_loss

    def postprocess(self, pred, target=None):
        if target is None:
            raise ValueError('target is required for postprocessing.')

        img = target[0]  # (B, 3, H, W)
        mask = target[1]  # (B, C, H, W)

        pred = F.interpolate(pred, (img.size(2), img.size(3))).detach()
        # final_pred = []
        # for p, i in zip(pred, img):
        #     p = np.transpose(p.cpu().numpy(), (1, 2, 0))
        #     i = np.array(TF.to_pil_image(i.cpu()))
        #     final_pred.append(dense_crf(p, i))

        # return torch.tensor(final_pred).to(pred.device).argmax(dim=1), mask.argmax(dim=1)

        return pred.argmax(dim=1), mask.argmax(dim=1)

    def save_checkpoint(self, ckpt_path, **kwargs):
        """Save model checkpoint."""

        torch.save({
            'model_state_dict': self.state_dict(),
            **kwargs,
        }, ckpt_path)
