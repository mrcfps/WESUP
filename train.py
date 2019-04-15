import argparse
import os
import warnings

warnings.filterwarnings('ignore')

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import ToTensor
from torchvision.models import vgg13

import config
from utils.data import SuperpixelDataset
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
    dataset = SuperpixelDataset(os.path.join(args.dataset_path, 'train'))
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=os.cpu_count() // 2)

    vgg = vgg13(pretrained=True)
    wessup = Wessup(vgg.features, device=device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(wessup.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(args.epochs):
        print('Epoch {}/{}'.format(epoch + 1, args.epochs))
        print('-' * 10)

        for img, sp_maps, sp_labels in tqdm(dataloader):
            img = img.to(device)
            sp_maps = sp_maps.to(device)
            sp_labels = sp_labels.to(device)

            optimizer.zero_grad()
            sp_pred = wessup(img, sp_maps.squeeze())
            loss = criterion(sp_pred, sp_labels.squeeze())
            loss.backward()
            optimizer.step()

        print('Loss: {}\n'.format(loss.item()))
