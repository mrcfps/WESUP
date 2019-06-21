from abc import ABC, abstractmethod

import torch.nn as nn


class BaseModel(ABC, nn.Module):
    """A base model class.

    It should satisfy following APIs:

    >>> model = DerivedBaseModel()
    >>> input_, target = model.preprocess(*data)
    >>> pred = model(input_)
    >>> loss = model.compute_loss(pred, target)
    >>> pred, target = model.postprocess(pred, target)
    >>> metrics = model.evaluate(pred, target)
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
    def postprocess(self, pred, target):
        """Postprocess raw prediction and target before calling evaluate method."""

    def evaluate(self, pred, target, metric_funcs):
        """Running several metrics to evaluate model performance."""

        metrics = {}

        for func in metric_funcs:
            metrics[func.__name__] = func(pred, target)

        return metrics
