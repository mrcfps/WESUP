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
from utils.metrics import detection_f1
from utils.metrics import object_dice
from utils.metrics import object_hausdorff
from utils.history import HistoryTracker
from utils.preprocessing import preprocess_superpixels
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
    parser.add_argument('-b', '--backbone', default='vgg16',
                        help='CNN backbone to use (such as vgg13, resnet50 and densenet121)')
    parser.add_argument('-r', '--resume-ckpt',
                        help='Path to previous checkpoint for resuming training')
    parser.add_argument('--lr', type=float, default=0.001,
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
    labeled_samples = torch.sum(y_true.sum(dim=1) > 0).float()

    if labeled_samples.item() == 0:
        return torch.tensor(0.)

    ce = -y_true * torch.log(y_hat)
    if weight is not None:
        ce = ce * weight.unsqueeze(0)

    return torch.sum(ce) / labeled_samples


def train_one_iteration(model, optimizer, phase, *data):
    img, segments, mask = data

    img = img.to(device)
    segments = segments.to(device).squeeze()
    mask = mask.to(device).squeeze()
    sp_maps, sp_labels = preprocess_superpixels(segments, mask)

    optimizer.zero_grad()
    metrics = dict()

    # weighted cross entropy
    ce = partial(cross_entropy, weight=torch.Tensor(config.CLASS_WEIGHTS).to(device))

    with torch.set_grad_enabled(phase == 'train'):
        sp_pred = model(img, sp_maps)

        # total number of superpixels
        total_num = sp_pred.size(0)

        # number of labeled superpixels
        labeled_num = sp_labels.size(0)

        if labeled_num < total_num:
            # weakly-supervised mode
            metrics['labeled_sp_ratio'] = labeled_num / total_num
            propagated_labels = label_propagate(model.clf_input_features, sp_labels,
                                                config.PROPAGATE_THRESHOLD)
            metrics['propagated_labels'] = propagated_labels.sum().item()
            loss = ce(sp_pred[:labeled_num], sp_labels)
            propagate_loss = config.PROPAGATE_WEIGHT * ce(sp_pred[labeled_num:], propagated_labels)
            metrics['propagate_loss'] = propagate_loss.item()
            loss += propagate_loss
        else:  # fully-supervised mode
            loss = ce(sp_pred, sp_labels)

        metrics['loss'] = loss.item()
        if phase == 'train':
            loss.backward()
            optimizer.step()

    mask = mask.argmax(dim=-1)
    pred_mask = compute_mask_with_superpixel_prediction(sp_pred, sp_maps)
    pred_mask = pred_mask.argmax(dim=0)
    metrics['pixel_acc'] = accuracy(pred_mask, mask)
    metrics['dice'] = dice(pred_mask, mask)

    # calculate object-level metrics in validation phase
    if phase == 'val':
        metrics['detection_f1'] = detection_f1(pred_mask, mask)
        metrics['object_dice'] = object_dice(pred_mask, mask)
        metrics['object_hausdorff'] = object_hausdorff(pred_mask, mask)

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
        checkpoint = torch.load(args.resume_ckpt, map_location=device)

        # load previous model states
        backbone = checkpoint['backbone']
        wessup = Wessup(backbone)
        wessup.load_state_dict(checkpoint['model_state_dict'])
    else:  # train a new model
        record_dir = record.prepare_record_dir()
        record.copy_source_files(record_dir)

        # create new model
        wessup = Wessup(args.backbone)

    wessup.to(device)
    tracker = HistoryTracker(os.path.join(record_dir, 'history.csv'))
    record.save_params(record_dir, args)

    if args.warmup > 0:
        # only optimize classifier of wessup
        optimizer = optim.SGD(
            wessup.classifier.parameters(),
            lr=0.005,
            momentum=config.MOMENTUM,
            weight_decay=config.WEIGHT_DECAY
        )

        print('\nWarmup Stage')
        print('=' * 20)
        for epoch in range(1, args.warmup + 1):
            print('\nWarmup epoch {}/{}'.format(epoch, args.warmup))
            print('-' * 10)
            train_one_epoch(wessup, optimizer, epoch)

    if args.resume_ckpt is not None:
        # load previous optimizer states and set learning rate to given value
        optimizer = optim.SGD(
            wessup.parameters(),
            lr=args.lr
        )
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        optimizer.param_groups[0]['lr'] = args.lr

        initial_epoch = checkpoint['epoch'] + 1
    else:
        optimizer = optim.SGD(
            wessup.parameters(),
            lr=args.lr,
            momentum=config.MOMENTUM,
            weight_decay=config.WEIGHT_DECAY
        )
        initial_epoch = 1

    total_epochs = args.epochs + initial_epoch - 1

    if not args.no_lr_decay:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max',
                                                         factor=0.5, min_lr=1e-7,
                                                         verbose=True)

    print('\nTraining Stage')
    print('=' * 20)

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

    tracker.report()
