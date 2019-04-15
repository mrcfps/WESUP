import argparse
import os
import warnings

warnings.filterwarnings('ignore')

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision.models import vgg13

import config
from utils import predict_whole_patch
from utils.data import SuperpixelDataset
from utils.metrics import superpixel_accuracy
from utils.metrics import pixel_accuracy
from utils.metrics import MetricsTracker
from wessup import Wessup


def build_cli_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_path', help='Path to dataset')
    parser.add_argument('--gpu', action='store_true', default=False,
                        help='Whether to use gpu')
    parser.add_argument('-e', '--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('-m', '--message', help='Note on this experiment')

    return parser


if __name__ == '__main__':
    parser = build_cli_parser()
    args = parser.parse_args()

    device = 'cuda' if args.gpu else 'cpu'
    datasets = {
        'train': SuperpixelDataset(args.dataset_path, train=True, to_device=device),
        'val': SuperpixelDataset(args.dataset_path, train=False, to_device=device),
    }
    dataloaders = {
        'train': DataLoader(datasets['train'], batch_size=1,
                            shuffle=True, num_workers=os.cpu_count() // 2),
        'val': DataLoader(datasets['val'], batch_size=1,
                          shuffle=True, num_workers=os.cpu_count() // 2),
    }

    vgg = vgg13(pretrained=True)
    wessup = Wessup(vgg.features, device=device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(wessup.parameters(), lr=0.001, momentum=0.9)
    tracker = MetricsTracker('history.csv')
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau()

    for epoch in range(args.epochs):
        print('Epoch {}/{}'.format(epoch + 1, args.epochs))
        print('-' * 10)

        for phase in ['train', 'val']:
            print(f'{phase.capitalize()} phase:')

            if phase == 'train':
                wessup.train()
                tracker.train()
            else:
                wessup.eval()
                tracker.eval()

            for img, mask, sp_maps, sp_labels in tqdm(dataloaders[phase]):
                # squeeze out `n_samples` dimension
                mask.squeeze_()
                sp_maps.squeeze_()
                sp_labels.squeeze_()

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    sp_pred = wessup(img, sp_maps)
                    loss = criterion(sp_pred, sp_labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                sp_acc = superpixel_accuracy(sp_pred, sp_labels)
                pred_mask = predict_whole_patch(sp_pred, sp_labels, sp_maps)
                pixel_acc = pixel_accuracy(pred_mask, mask)

                tracker.step(loss=loss.item(), sp_acc=sp_acc.item(),
                             pixel_acc=pixel_acc.item())

            tracker.log()
