from abc import ABC, abstractmethod

import torch.nn as nn

from utils.metrics import accuracy
from utils.metrics import dice
from utils.metrics import detection_f1
from utils.metrics import object_dice
from utils.metrics import object_hausdorff


class BaseModel(ABC, nn.Module):
    """A base model class.

    It should satisfy following APIs:

    >>> model = DerivedBaseModel()
    >>> input_, target = model.preprocess(*data)
    >>> pred = model(input_)
    >>> loss = model.compute_loss(pred, target)
    >>> model.save_checkpoint('/path/to/ckpt')
    """

    @abstractmethod
    def get_default_dataset(self, root_dir, train=True):
        """Get default dataset for training/validation.

        Args:
            root_dir: path to dataset root
            train: whether it is a dataset for training.
        """

    @abstractmethod
    def get_default_optimizer(self, checkpoint=None):
        """Get default optimizer for training.

        Args:
            checkpoint (optional): checkpoint to recover optimizer state

        Returns:
            optimizer: default model optimizer
            scheduler (optional): default learning rate scheduler
        """

    @abstractmethod
    def preprocess(self, *data):
        """Preprocess data from dataloaders and return model inputs and targets."""

    @abstractmethod
    def compute_loss(self, pred, target, metrics=None):
        """Compute objective function."""

    @abstractmethod
    def save_checkpoint(self, ckpt_path, **kwargs):
        """Save model checkpoint."""

    @abstractmethod
    def _pre_evaluate_hook(self, pred, target):
        """Hook function to run before calling evaluate method."""

    def evaluate(self, pred, target):
        """Running several metrics to evaluate model performance."""

        pred, target = self._pre_evaluate_hook(pred, target)
        metrics = {
            'pixel_acc': accuracy(pred, target),
            'dice': dice(pred, target),
        }

        # calculate object-level metrics in validation phase
        if not self.training:
            metrics['detection_f1'] = detection_f1(pred, target)
            metrics['object_dice'] = object_dice(pred, target)
            # metrics['object_hausdorff'] = object_hausdorff(pred, target)

        return metrics
