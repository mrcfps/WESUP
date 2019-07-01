"""
Training module.
"""

import argparse
import os
import warnings
from shutil import rmtree

import numpy as np
from tqdm import tqdm

import torch

from models import Wessup
from models import CDWS
from utils import record
from utils import log
from utils.metrics import accuracy
from utils.metrics import dice
from utils.metrics import detection_f1
from utils.metrics import object_dice
from utils.history import HistoryTracker

warnings.filterwarnings('ignore')

# which device to use
device = None

# Train/val dataLoaders dictionary
dataloaders = None

# path to experiment record directory
record_dir = None

# history metrics tracker object
tracker = None


def build_cli_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_path', help='Path to dataset')
    parser.add_argument('-d', '--device', default=('cuda' if torch.cuda.is_available() else 'cpu'),
                        help='Which device to use')
    parser.add_argument('-m', '--model', default='wessup', choices=['wessup', 'cdws'],
                        help='Which model to use')
    parser.add_argument('-e', '--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('-b', '--batch-size', type=int, default=1, help='Minibatch size')
    parser.add_argument('-j', '--jobs', type=int, default=os.cpu_count(),
                        help='Number of CPUs to use for data preparation')
    parser.add_argument('--ckpt-period', type=int, default=10,
                        help='Period for saving model checkpoint')
    parser.add_argument('-r', '--resume-ckpt',
                        help='Path to previous checkpoint for resuming training')
    parser.add_argument('--message', help='Note on this experiment')
    parser.add_argument('--smoke', action='store_true', default=False,
                        help='Whether this is a smoke test')

    return parser


def train_one_iteration(model, optimizer, phase, *data):
    data = [datum.to(device) for datum in data]
    input_, target = model.preprocess(*data)

    optimizer.zero_grad()
    metrics = dict()

    with torch.set_grad_enabled(phase == 'train'):
        pred = model(input_)
        if phase == 'train':
            loss = model.compute_loss(pred, target, metrics=metrics)
            metrics['loss'] = loss.item()

            loss.backward()
            optimizer.step()

    metric_funcs = [accuracy, dice]
    if phase == 'val':
        metric_funcs.extend([detection_f1, object_dice])

    pred, target = model.postprocess(pred, target)
    tracker.step({**metrics, **model.evaluate(pred, target, metric_funcs)})


def train_one_epoch(model, optimizer, no_val=False):
    phases = ['train'] if no_val else ['train', 'val']
    for phase in phases:
        print(f'{phase.capitalize()} phase:')

        if phase == 'train':
            model.train()
            tracker.train()
        else:
            model.eval()
            tracker.eval()

        pbar = tqdm(dataloaders[phase])
        for data in pbar:
            train_one_iteration(model, optimizer, phase, *data)

        pbar.write(tracker.log())
        pbar.close()


def fit(args):
    ############################# MODEL #############################
    checkpoint = None
    if args.resume_ckpt is not None:
        print(f'Loading checkpoints from {args.resume_ckpt}.')
        checkpoint = torch.load(args.resume_ckpt, map_location=device)

    if args.model == 'wessup':
        model = Wessup(checkpoint=checkpoint)
    elif args.model == 'cdws':
        model = CDWS(checkpoint=checkpoint)
    else:
        raise ValueError(f'Unsupported model: {args.model}')

    model = model.to(device)
    record.save_params(record_dir,
                       {**vars(args), 'model_config': model.get_default_config()})

    ############################# DATA #############################
    global dataloaders
    dataloaders = {
        'train': torch.utils.data.DataLoader(
            model.get_default_dataset(os.path.join(args.dataset_path, 'train')),
            batch_size=args.batch_size, shuffle=True, num_workers=args.jobs
        )
    }

    val_path = os.path.join(args.dataset_path, 'val')
    if os.path.exists(val_path):
        dataloaders['val'] = torch.utils.data.DataLoader(
            model.get_default_dataset(val_path, train=False),
            batch_size=1, num_workers=args.jobs
        )

    ############################# TRAIN #############################
    log('\nTraining Stage', '=')
    optimizer, scheduler = model.get_default_optimizer(checkpoint)
    initial_epoch = checkpoint['epoch'] + 1 if checkpoint is not None else 1
    total_epochs = args.epochs + initial_epoch - 1

    for epoch in range(initial_epoch, total_epochs + 1):
        log('\nEpoch {}/{}'.format(epoch, total_epochs), '-')

        tracker.start_new_epoch(optimizer.param_groups[0]['lr'])
        train_one_epoch(model, optimizer, no_val=(not os.path.exists(val_path)))

        if scheduler is not None:
            scheduler.step(np.mean(tracker.history['loss']))

        # save metrics to csv file
        tracker.save()

        # save learning curves
        record.plot_learning_curves(tracker.save_path)

        if epoch % args.ckpt_period == 0:
            # save checkpoints for resuming training
            ckpt_path = os.path.join(
                record_dir, 'checkpoints', 'ckpt.{:04d}.pth'.format(epoch))
            model.save_checkpoint(ckpt_path, epoch=epoch,
                                  optimizer_state_dict=optimizer.state_dict())

    tracker.report()


if __name__ == '__main__':
    parser = build_cli_parser()
    args = parser.parse_args()

    device = args.device

    if args.resume_ckpt is not None:
        record_dir = os.path.dirname(os.path.dirname(args.resume_ckpt))
    else:
        record_dir = record.prepare_record_dir()
        record.copy_source_files(record_dir)

    tracker = HistoryTracker(os.path.join(record_dir, 'history.csv'))

    try:
        fit(args)
    except KeyboardInterrupt:
        # stop training via Ctrl+C
        pass
    except:
        rmtree(record_dir, ignore_errors=True)
        raise
    finally:
        if args.smoke:
            rmtree(record_dir, ignore_errors=True)
