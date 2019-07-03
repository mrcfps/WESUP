from abc import ABC, abstractmethod
from collections import defaultdict

import torch.nn as nn
import numpy as np
from tqdm import tqdm


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
    def get_default_config(self):
        """Get default model configurations."""

    @abstractmethod
    def get_default_dataset(self, root_dir, train=True, proportion=1.0):
        """Get default dataset for training/validation.

        Args:
            root_dir: path to dataset root
            train: whether it is a dataset for training
            proportion: proportion of data to be used

        Returns:
            dataset: a `torch.utils.data.Dataset` instance
        """

    @abstractmethod
    def get_default_optimizer(self, checkpoint=None):
        """Get default optimizer for training.

        Args:
            checkpoint: checkpoint to recover optimizer state

        Returns:
            optimizer: default model optimizer
            scheduler: default learning rate scheduler (could be `None`)
        """

    @abstractmethod
    def preprocess(self, *data):
        """Preprocess data from dataloaders and return model inputs and targets.

        Args:
            *data: data returned from dataloaders

        Returns:
            input: input to feed into the model of size (B, H, W)
            target: desired output (or any additional information) to compute loss
                and evaluate performance
        """

    @abstractmethod
    def compute_loss(self, pred, target, metrics=None):
        """Compute objective function.

        Args:
            pred: model prediction from the `forward` step
            target: target computed from `preprocess` method

        Returns:
            loss: model loss
        """

    @abstractmethod
    def save_checkpoint(self, ckpt_path, **kwargs):
        """Save model checkpoint.

        Args:
            ckpt_path: path to checkpoint to be saved
            kwargs: additional information to be included in the checkpoint object
        """

    @abstractmethod
    def postprocess(self, pred, target=None):
        """Postprocess raw prediction and target before calling `evaluate` method.

        Args:
            pred: prediction computed from the `forward` step
            target: target computed from `preprocess` method (optional)

        Returns:
            pred: postprocessed prediction
            target: postprocessed target
        """

    def evaluate(self, pred, target, metric_funcs, verbose=False):
        """Running several metrics to evaluate model performance.

        Args:
            pred: prediction of size (B, H, W), either torch.Tensor or numpy array
            target: ground truth of size (B, H, W), either torch.Tensor or numpy array
            metric_funcs: list of metric functions
            verbose: whether to show progress bar

        Returns:
            metrics: a dictionary containing all metrics
        """

        metrics = defaultdict(list)

        iterable = zip(pred, target)
        if verbose:
            iterable = tqdm(iterable, total=len(pred))

        for P, G in iterable:
            for func in metric_funcs:
                metrics[func.__name__].append(func(P, G))

        return {k: np.mean(v) for k, v in metrics.items()}
