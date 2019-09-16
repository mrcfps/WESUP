import os
import logging
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm

from utils import underline, record
from utils.history import HistoryTracker


class BaseConfig:
    """A base model configuration class."""

    # batch size for training
    batch_size = 1

    # number of epochs for training
    epochs = 10

    # numerical stability term
    epsilon = 1e-7

    def __str__(self):
        return '\n'.join(f'{attr:<32s}{getattr(self, attr)}'
                         for attr in dir(self) if not attr.startswith('_'))

    def to_dict(self):
        return {
            attr: getattr(self, attr) for attr in dir(self)
            if not attr.startswith('_') and attr != 'to_dict'
        }


class BaseTrainer(ABC):
    """A base trainer class."""

    def __init__(self, model, **kwargs):
        """Initialize a BaseTrainer.

        Args:
            model: a model for training (should be a `torch.nn.Module`)
            kwargs (optional): additional configuration

        Returns:
            trainer: a new BaseTrainer instance
        """

        self.device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.kwargs = kwargs

        # Initialize logger.
        if kwargs.get('logger'):
            self.logger = kwargs.get('logger')
        else:
            self.logger = logging.getLogger('Train')
            self.logger.setLevel(logging.DEBUG)
            self.logger.addHandler(logging.StreamHandler())

        # Training components.
        self.initial_epoch = 1
        self.record_dir = None
        self.tracker = HistoryTracker()
        self.dataloaders = None
        self.optimizer, self.scheduler = self.get_default_optimizer()
        self.metric_funcs = []

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

    def get_default_optimizer(self):
        """Get default optimizer for training.

        Returns:
            optimizer: default model optimizer
            scheduler: default learning rate scheduler (could be `None`)
        """

        return torch.optim.SGD(self.model.parameters(), lr=1e-3), None

    def preprocess(self, *data):
        """Preprocess data from dataloaders and return model inputs and targets.

        Args:
            *data: data returned from dataloaders

        Returns:
            input: input to feed into the model of size (B, H, W)
            target: desired output (or any additional information) to compute loss
                and evaluate performance
        """

        return [datum.to(self.device) for datum in data]

    @abstractmethod
    def compute_loss(self, pred, target, metrics=None):
        """Compute objective function.

        Args:
            pred: model prediction from the `forward` step
            target: target computed from `preprocess` method
            metrics: dict for tracking metrics when computing loss

        Returns:
            loss: model loss
        """

    def load_checkpoint(self, ckpt_path=None):
        """Load checkpointed model weights, optimizer states, etc, from given path.

        Args:
            ckpt_path: path to checkpoint
        """

        if ckpt_path is not None:
            self.record_dir = Path(ckpt_path).parent.parent
            self.logger.info(f'Loading checkpoint from {ckpt_path}.')
            checkpoint = torch.load(ckpt_path, map_location=self.device)

            self.initial_epoch = checkpoint['epoch'] + 1
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            if self.scheduler is not None:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        else:
            self.record_dir = Path(record.prepare_record_dir())
            record.copy_source_files(self.record_dir)

    def save_checkpoint(self, ckpt_path, **kwargs):
        """Save model checkpoint.

        Args:
            ckpt_path: path to checkpoint to be saved
            kwargs: additional information to be included in the checkpoint object
        """

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            **kwargs,
        }, ckpt_path)

    def postprocess(self, pred, target=None):
        """Postprocess raw prediction and target before calling `evaluate` method.

        Args:
            pred: prediction computed from the `forward` step
            target: target computed from `preprocess` method (optional)

        Returns:
            pred: postprocessed prediction
            target: postprocessed target (optional)
        """

        if target is not None:
            return pred, target
        return pred

    def train_one_iteration(self, phase, *data):
        """Hook for training one iteration.

        Args:
            phase: either 'train' or 'val'
            *data: input data
        """

        input_, target = self.preprocess(*data)

        self.optimizer.zero_grad()
        metrics = dict()

        with torch.set_grad_enabled(phase == 'train'):
            pred = self.model(input_)
            if phase == 'train':
                loss = self.compute_loss(pred, target, metrics=metrics)
                metrics['loss'] = loss.item()

                loss.backward()
                self.optimizer.step()

        pred, target = self.postprocess(pred, target)
        self.tracker.step({**metrics, **self.evaluate(pred, target)})
    
    def train_one_epoch(self, no_val=False):
        """Hook for training one epoch.

        Args:
            no_val: whether to disable validation
        """

        phases = ['train'] if no_val else ['train', 'val']
        for phase in phases:
            self.logger.info(f'{phase.capitalize()} phase:')
            start = time.time()

            if phase == 'train':
                self.model.train()
                self.tracker.train()
            else:
                self.model.eval()
                self.tracker.eval()

            pbar = tqdm(self.dataloaders[phase])
            for data in pbar:
                try:
                    self.train_one_iteration(phase, *data)
                except RuntimeError as ex:
                    self.logger.exception(ex)

            self.logger.info(f'Took {time.time() - start:.2f}s.')
            self.logger.info(self.tracker.log())
            pbar.close()

    def post_epoch_hook(self, epoch):
        """Hook for post-epoch stage.

        Args:
            epoch: current epoch
        """

        pass

    def train(self, data_root, **kwargs):
        """Start training process.

        Args:
            data_root: path to dataset, should contain a subdirectory named 'train'
                (and optionally 'val')

        Kwargs (optional):
            metrics: a list of functions for computing metrics
            checkpoint: path to checkpoint for resuming training
            epochs: number of epochs for training
            batch_size: mini-batch size for training
            proportion: proportion of training data to be used
        """

        # Merge configurations.
        self.kwargs = {**self.kwargs, **kwargs}

        self.load_checkpoint(self.kwargs.get('checkpoint'))
        self.logger.addHandler(logging.FileHandler(self.record_dir / 'train.log'))
        serializable_kwargs = {
            k: v for k, v in self.kwargs.items()
            if isinstance(v, (int, float, str, tuple))
        }
        record.save_params(self.record_dir, serializable_kwargs)
        self.logger.info(str(serializable_kwargs) + '\n')
        self.tracker.save_path = self.record_dir / 'history.csv'
        data_root = Path(data_root)
        train_path = data_root / 'train'
        val_path = data_root / 'val'
        train_dataset = self.get_default_dataset(train_path,
                                                 proportion=self.kwargs.get('proportion', 1))
        train_dataset.summary(logger=self.logger)

        self.dataloaders = {
            'train': torch.utils.data.DataLoader(
                train_dataset,
                batch_size=self.kwargs.get('batch_size'),
                shuffle=True, num_workers=os.cpu_count())
        }

        if val_path.exists():
            val_dataset = self.get_default_dataset(val_path, train=False)
            val_dataset.summary(logger=self.logger)
            self.dataloaders['val'] = torch.utils.data.DataLoader(
                val_dataset, batch_size=1,
                num_workers=os.cpu_count())

        self.logger.info(underline('\nTraining Stage', '='))
        self.metric_funcs = self.kwargs.get('metrics')

        epochs = self.kwargs.get('epochs')
        total_epochs = epochs + self.initial_epoch - 1

        for epoch in range(self.initial_epoch, total_epochs + 1):
            self.logger.info(underline('\nEpoch {}/{}'.format(epoch, total_epochs), '-'))

            self.tracker.start_new_epoch(self.optimizer.param_groups[0]['lr'])
            self.train_one_epoch(no_val=(not val_path.exists()))
            self.post_epoch_hook(epoch)

            # save metrics to csv file
            self.tracker.save()

            # save learning curves
            record.plot_learning_curves(self.tracker.save_path)

            # save checkpoints for resuming training
            ckpt_path = self.record_dir / 'checkpoints' / f'ckpt.{epoch:04d}.pth'
            self.save_checkpoint(ckpt_path, epoch=epoch,
                                 optimizer_state_dict=self.optimizer.state_dict())

            # remove previous checkpoints
            for ckpt_path in sorted((self.record_dir / 'checkpoints').glob('*.pth'))[:-1]:
                os.remove(ckpt_path)

        self.logger.info(self.tracker.report())

    def evaluate(self, pred, target, verbose=False):
        """Running several metrics to evaluate model performance.

        Args:
            pred: prediction of size (B, H, W), either torch.Tensor or numpy array
            target: ground truth of size (B, H, W), either torch.Tensor or numpy array
            verbose: whether to show progress bar

        Returns:
            metrics: a dictionary containing all metrics
        """

        metrics = defaultdict(list)

        iterable = zip(pred, target)
        if verbose:
            iterable = tqdm(iterable, total=len(pred))

        for P, G in iterable:
            for func in self.metric_funcs:
                metrics[func.__name__].append(func(P, G))

        return {k: np.mean(v) for k, v in metrics.items()}
