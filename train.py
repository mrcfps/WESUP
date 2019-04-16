import argparse
import os
import warnings

warnings.filterwarnings('ignore')

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import vgg13

from wessup import Wessup
from utils import record
from utils import predict_whole_patch
from utils.data import get_trainval_dataloaders
from utils.metrics import superpixel_accuracy
from utils.metrics import pixel_accuracy
from utils.history import HistoryTracker


def build_cli_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_path', help='Path to dataset')
    parser.add_argument('--gpu', action='store_true', default=False,
                        help='Whether to use gpu')
    parser.add_argument('-e', '--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('-j', '--jobs', type=int, default=int(os.cpu_count() / 2),
                        help='Number of CPUs to use for preparing superpixels')
    parser.add_argument('-r', '--resume-ckpt',
                        help='Path to previous checkpoint for resuming training')
    parser.add_argument('-m', '--message', help='Note on this experiment')

    return parser


if __name__ == '__main__':
    parser = build_cli_parser()
    args = parser.parse_args()

    device = 'cuda' if args.gpu else 'cpu'
    dataloaders = get_trainval_dataloaders(args.dataset_path, args.jobs)

    if args.resume_ckpt is not None:
        record_dir = os.path.dirname(os.path.dirname(args.resume_ckpt))
        checkpoint = torch.load(args.resume_ckpt)
        wessup = Wessup(vgg13().features, device=device)
        wessup.load_state_dict(checkpoint['model_state_dict'])
        initial_epoch = checkpoint['epoch'] + 1
    else:
        record_dir = record.prepare_record_dir()
        wessup = Wessup(vgg13(pretrained=True).features, device=device)
        initial_epoch = 0

    sp_feature_length = wessup.extractor.sp_feature_length
    print(f'Wessup with {type(wessup.extractor)} backend ({sp_feature_length} superpixel features).')

    record.save_params(record_dir, args)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(wessup.parameters(), lr=0.001, momentum=0.9)

    history_path = os.path.join(record_dir, 'history.csv')
    tracker = HistoryTracker(history_path)

    total_epochs = args.epochs + initial_epoch
    for epoch in range(initial_epoch, total_epochs):
        print('\nEpoch {}/{}'.format(epoch + 1, total_epochs))
        print('-' * 10)

        for phase in ['train', 'val']:
            print(f'{phase.capitalize()} phase:')

            if phase == 'train':
                wessup.train()
                tracker.train()
            else:
                wessup.eval()
                tracker.eval()

            pbar = tqdm(dataloaders[phase])
            for img, mask, sp_maps, sp_labels in pbar:
                img = img.to(device)
                mask = mask.to(device)
                sp_maps = sp_maps.to(device)
                sp_labels = sp_labels.to(device)

                # squeeze out `n_samples` dimension
                mask.squeeze_()
                sp_maps.squeeze_()
                sp_labels.squeeze_()

                optimizer.zero_grad()
                metrics = dict()

                with torch.set_grad_enabled(phase == 'train'):
                    sp_pred = wessup(img, sp_maps)
                    loss = criterion(sp_pred, sp_labels)
                    metrics['loss'] = loss.item()
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                pred_mask = predict_whole_patch(sp_pred, sp_maps)
                metrics['sp_acc'] = superpixel_accuracy(sp_pred, sp_labels).item()
                metrics['pixel_acc'] = pixel_accuracy(pred_mask, mask).item()

                pbar.set_postfix_str(tracker.step(metrics))

            pbar.write(tracker.log())
            pbar.close()

        tracker.save()
        tracker.clear()

        # save learning curves
        record.plot_learning_curves(history_path)

        # save checkpoints for resume training
        torch.save({
            'epoch': epoch,
            'model_state_dict': wessup.state_dict(),
        }, os.path.join(record_dir, 'checkpoints', 'ckpt.{:04d}.pth'.format(epoch)))
