"""
Training module.
"""

import argparse
import os
import warnings

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.models import vgg13

import config
from wessup import Wessup
from utils import record
from utils.data import get_trainval_dataloaders
from utils.metrics import accuracy
from utils.metrics import dice
from utils.history import HistoryTracker
from infer import test_whole_images
from infer import compute_mask_with_superpixel_prediction

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
    parser.add_argument('--no-gpu', action='store_true', default=False,
                        help='Whether to avoid using gpu')
    parser.add_argument('-e', '--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('-w', '--warmup', type=int, default=0,
                        help='Number of warmup epochs (freeze CNN) before training')
    parser.add_argument('-j', '--jobs', type=int, default=int(os.cpu_count() / 2),
                        help='Number of CPUs to use for preparing superpixels')
    parser.add_argument('-b', '--backbone', default='vgg13',
                        help='CNN backbone to use (such as vgg13, resnet50 and densenet121)')
    parser.add_argument('-r', '--resume-ckpt',
                        help='Path to previous checkpoint for resuming training')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate for optimizer')
    parser.add_argument('-m', '--message', help='Note on this experiment')

    return parser


def train_one_iteration(model, optimizer, phase, *data):
    if len(data) == 4:  # data from `FullAnnotationDataset`
        img, mask, sp_maps, sp_labels = data
    else:  # data from `DotAnnotationDataset`
        img, sp_maps, sp_labels = data
        mask = None

    img = img.to(device)
    sp_maps = sp_maps.to(device).squeeze()
    sp_labels = sp_labels.to(device).squeeze()

    optimizer.zero_grad()
    metrics = dict()

    with torch.set_grad_enabled(phase == 'train'):
        sp_pred = model(img, sp_maps)
        if sp_pred.size(0) > sp_labels.size(0):
            # weakly-supervised mode
            sp_pred = sp_pred[:sp_labels.size(0)]
            loss = F.cross_entropy(sp_pred, sp_labels,
                                   weight=torch.Tensor(config.CLASS_WEIGHTS).to(device))
        else:  # fully-supervised mode
            loss = F.cross_entropy(sp_pred, sp_labels,
                                   weight=torch.Tensor(config.CLASS_WEIGHTS).to(device))
        metrics['loss'] = loss.item()
        if phase == 'train':
            loss.backward()
            optimizer.step()

    metrics['sp_acc'] = accuracy(sp_pred.argmax(dim=-1), sp_labels)

    if mask is not None:
        mask = mask.to(device).squeeze()
        pred_mask = compute_mask_with_superpixel_prediction(sp_pred, sp_maps)
        metrics['pixel_acc'] = accuracy(pred_mask, mask)
        metrics['dice'] = dice(pred_mask, mask)

    tracker.step(metrics)


def train_one_epoch(model, optimizer, epoch, warmup=False):
    for phase in ['train', 'val']:
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

    if not warmup:
        # save metrics to csv file
        tracker.save()

        # save learning curves
        record.plot_learning_curves(tracker.save_path)

        if epoch % config.CHECKPOINT_PERIOD == 0:
            # save checkpoints for resuming training
            ckpt_path = os.path.join(
                record_dir, 'checkpoints', 'ckpt.{:04d}.pth'.format(epoch))
            torch.save({
                'epoch': epoch,
                'backbone': wessup.backbone_name,
                'model_state_dict': wessup.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, ckpt_path)
            print(f'Save checkpoint to {ckpt_path}.')

    tracker.clear()


if __name__ == '__main__':
    parser = build_cli_parser()
    args = parser.parse_args()

    device = 'cpu' if args.no_gpu and not torch.cuda.is_available() else 'cuda'
    dataloaders = get_trainval_dataloaders(args.dataset_path, args.jobs)

    if args.resume_ckpt is not None:
        record_dir = os.path.dirname(os.path.dirname(args.resume_ckpt))
        tracker = HistoryTracker(os.path.join(record_dir, 'history.csv'))

        checkpoint = torch.load(args.resume_ckpt)

        # load previous model states
        backbone = checkpoint['backbone']
        wessup = Wessup(backbone)
        wessup.load_state_dict(checkpoint['model_state_dict'])
        wessup.to(device)

        # load previous optimizer states and set learning rate to given value
        optimizer = optim.SGD(
            wessup.parameters(),
            lr=args.lr
        )
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        optimizer.param_groups[0]['lr'] = args.lr

        initial_epoch = checkpoint['epoch'] + 1
    else:  # train a new model
        record_dir = record.prepare_record_dir()
        record.copy_source_files(record_dir)

        tracker = HistoryTracker(os.path.join(record_dir, 'history.csv'))

        # create new model
        wessup = Wessup(args.backbone)
        wessup.to(device)

        if args.warmup > 0:
            # only optimize classifier of wessup
            optimizer = optim.SGD(
                wessup.classifier.parameters(),
                lr=0.01,
                momentum=config.MOMENTUM,
                weight_decay=config.WEIGHT_DECAY
            )

            print('\nWarmup Stage')
            print('=' * 20)
            for epoch in range(1, args.warmup + 1):
                print('\nWarmup epoch {}/{}'.format(epoch, args.warmup))
                print('-' * 10)
                train_one_epoch(wessup, optimizer, epoch, warmup=True)

        optimizer = optim.SGD(
            wessup.parameters(),
            lr=args.lr,
            momentum=config.MOMENTUM,
            weight_decay=config.WEIGHT_DECAY
        )
        initial_epoch = 1

    record.save_params(record_dir, args)

    print('\nTraining Stage')
    print('=' * 20)
    total_epochs = args.epochs + initial_epoch - 1

    for epoch in range(initial_epoch, total_epochs + 1):
        print('\nEpoch {}/{}'.format(epoch, total_epochs))
        print('-' * 10)
        train_one_epoch(wessup, optimizer, epoch)

        # test on whole images
        if epoch % config.WHOLE_IMAGE_TEST_PERIOD == 0:
            viz_dir = os.path.join(record_dir, 'viz')
            test_whole_images(
                wessup, os.path.join(args.dataset_path, 'val-whole'),
                viz_dir=viz_dir, epoch=epoch, device=device, num_workers=args.jobs
            )
