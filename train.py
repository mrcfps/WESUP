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
import torch.optim as optim

import config
from models import Wessup
from models import CDWS
from utils import record
from utils import log
from utils.history import HistoryTracker
from utils.data import get_trainval_dataloaders

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
    parser.add_argument('--model', default='wessup', choices=['wessup', 'cdws'],
                        help='Which model to use')
    parser.add_argument('--mode', default='point', choices=['point', 'area', 'mask'],
                        help='Supervision mode')
    parser.add_argument('-p', '--point-ratio', type=float, default=5e-5,
                        help='Ratio of annotated points and total pixels')
    parser.add_argument('-e', '--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('-w', '--warmup', type=int, default=0,
                        help='Number of warmup epochs (freeze CNN) before training')
    parser.add_argument('-j', '--jobs', type=int, default=int(os.cpu_count() / 2),
                        help='Number of CPUs to use for preparing superpixels')
    parser.add_argument('-b', '--backbone', default='vgg16',
                        help='CNN backbone to use (such as vgg13, resnet50 and densenet121)')
    parser.add_argument('-r', '--resume-ckpt',
                        help='Path to previous checkpoint for resuming training')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate for optimizer')
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
        loss = model.compute_loss(pred, target)
        metrics['loss'] = loss.item()
        if phase == 'train':
            loss.backward()
            optimizer.step()

    tracker.step({**metrics, **model.evaluate(pred, target)})


def train_one_epoch(model, optimizer, warmup=False):
    phases = ['train'] if warmup else ['train', 'val']
    for phase in phases:
        print(f'{phase.capitalize()} phase:')

        if phase == 'train':
            model.train()
            tracker.train()
        else:
            model.eval()
            tracker.eval()

        if warmup:
            model.backbone.eval()

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
        model = Wessup(args.backbone, checkpoint=checkpoint)
    elif args.model == 'cdws':
        model = CDWS(checkpoint=checkpoint)
    else:
        raise ValueError(f'Unsupported model: {args.model}')

    model = model.to(device)

    ############################# WARMUP #############################
    if args.warmup > 0 and args.model == 'wessup':
        # only optimize classifier of wessup
        optimizer = optim.SGD(
            model.classifier.parameters(),
            lr=0.005,
            momentum=config.MOMENTUM,
            weight_decay=config.WEIGHT_DECAY
        )

        log('\nWarmup Stage', '=')
        for epoch in range(1, args.warmup + 1):
            log('\nWarmup epoch {}/{}'.format(epoch, args.warmup), '-')
            train_one_epoch(model, optimizer, warmup=True)

    ############################ OPTIMIZER ############################
    if checkpoint is not None:
        # load previous optimizer states and set learning rate to given value
        optimizer = optim.SGD(
            model.parameters(),
            lr=args.lr
        )
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        optimizer.param_groups[0]['lr'] = args.lr

        initial_epoch = checkpoint['epoch'] + 1
    else:
        optimizer = optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=config.MOMENTUM,
            weight_decay=config.WEIGHT_DECAY
        )
        initial_epoch = 1

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max',
                                                     factor=0.5, min_lr=1e-7,
                                                     verbose=True)

    ############################# TRAIN #############################
    log('\nTraining Stage', '=')
    total_epochs = args.epochs + initial_epoch - 1

    for epoch in range(initial_epoch, total_epochs + 1):
        log('\nEpoch {}/{}'.format(epoch, total_epochs), '-')

        tracker.start_new_epoch(optimizer.param_groups[0]['lr'])
        train_one_epoch(model, optimizer)
        scheduler.step(np.mean(tracker.history['val_dice']))

        # save metrics to csv file
        tracker.save()

        # save learning curves
        record.plot_learning_curves(tracker.save_path)

        if epoch % config.CHECKPOINT_PERIOD == 0:
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
    dataloaders = get_trainval_dataloaders(
        args.dataset_path, mode=args.mode,
        point_ratio=args.point_ratio, num_workers=args.jobs)

    if args.resume_ckpt is not None:
        record_dir = os.path.dirname(os.path.dirname(args.resume_ckpt))
    else:
        record_dir = record.prepare_record_dir()
        record.copy_source_files(record_dir)

    tracker = HistoryTracker(os.path.join(record_dir, 'history.csv'))
    record.save_params(record_dir, args)

    try:
        fit(args)
    except:
        rmtree(record_dir, ignore_errors=True)
        raise
    finally:
        if args.smoke:
            rmtree(record_dir, ignore_errors=True)
