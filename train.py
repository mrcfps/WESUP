"""
Training module.
"""

import argparse
import os
import warnings
from functools import partial

import numpy as np
from tqdm import tqdm

import torch
import torch.optim as optim

import config
from wessup import Wessup
from utils import record
from utils.semi import label_propagate
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
    parser.add_argument('-d', '--device', default=('cuda' if torch.cuda.is_available() else 'cpu'),
                        help='Which device to use')
    parser.add_argument('--no-lr-decay', action='store_true', default=False,
                        help='Whether to disable automatic learning rate decay')
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


def cross_entropy(y_hat, y_true, weight=None):
    """Semi-supervised cross entropy loss function.

    Args:
        y_hat: prediction tensor with size (N, c), where c is the number of classes
        y_true: label tensor with size (N, c). A sample won't be counted into loss
            if its label is all zeros.
        weight: class weight tensor with size (c,)

    Returns:
        cross_entropy: cross entropy loss computed only on samples with labels
    """

    # clamp all elements to prevent numerical overflow/underflow
    y_hat = torch.clamp(y_hat, min=config.EPSILON, max=(1 - config.EPSILON))

    # number of samples with labels
    n_labels = torch.sum(y_true.sum(dim=1) > 0).float()

    if n_labels.item() == 0:
        return 0

    ce = -y_true * torch.log(y_hat)
    if weight is not None:
        ce = ce * weight.unsqueeze(0)

    return torch.sum(ce) / n_labels


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

    # weighted cross entropy
    ce = partial(cross_entropy, weight=torch.Tensor(config.CLASS_WEIGHTS).to(device))

    with torch.set_grad_enabled(phase == 'train'):
        sp_pred = model(img, sp_maps)
        if sp_pred.size(0) > sp_labels.size(0):
            # weakly-supervised mode
            propagated_labels = label_propagate(model.lp_input_features, sp_labels)
            n_l = sp_labels.size(0)
            loss = ce(sp_pred[:n_l], sp_labels) + config.PROPAGATE_WEIGHT * ce(sp_pred[n_l:], propagated_labels)
        else:  # fully-supervised mode
            loss = ce(sp_pred, sp_labels)
        metrics['loss'] = loss.item()
        if phase == 'train':
            loss.backward()
            optimizer.step()

    if mask is not None:
        mask = mask.to(device).squeeze()
        pred_mask = compute_mask_with_superpixel_prediction(sp_pred, sp_maps)
        metrics['pixel_acc'] = accuracy(pred_mask, mask)
        metrics['dice'] = dice(pred_mask, mask)

    tracker.step(metrics)


def train_one_epoch(model, optimizer):
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


if __name__ == '__main__':
    parser = build_cli_parser()
    args = parser.parse_args()

    device = args.device
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

    if not args.no_lr_decay:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max',
                                                         factor=0.5, min_lr=1e-5,
                                                         verbose=True)

    for epoch in range(initial_epoch, total_epochs + 1):
        print('\nEpoch {}/{}'.format(epoch, total_epochs))
        print('-' * 10)

        tracker.start_new_epoch(optimizer.param_groups[0]['lr'])
        train_one_epoch(wessup, optimizer)

        if not args.no_lr_decay:
            scheduler.step(np.mean(tracker.history['val_dice']))

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

        # test on whole images
        if epoch % config.WHOLE_IMAGE_TEST_PERIOD == 0:
            viz_dir = os.path.join(record_dir, 'viz')
            test_whole_images(
                wessup, os.path.join(args.dataset_path, 'val-whole'),
                viz_dir=viz_dir, epoch=epoch, evaluate=False, num_workers=args.jobs
            )
