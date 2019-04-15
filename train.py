import argparse

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import ToTensor
from torchvision.models import vgg13

import config
from data import SuperpixelDataset
from wessup import Wessup


if __name__ == '__main__':
    dataset = SuperpixelDataset('data/train')
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=True)

    vgg = vgg13(pretrained=True)
    wessup = Wessup(vgg.features, 'cpu')
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(wessup.parameters(), lr=0.001, momentum=0.9)

    for idx, data in enumerate(dataloader):
        img, sp_maps, sp_labels = data
        optimizer.zero_grad()
        sp_pred = wessup(img, sp_maps.squeeze())
        loss = criterion(sp_pred, sp_labels.squeeze())
        loss.backward()
        optimizer.step()

        print(f'[{idx}] loss: {loss.item()}')

        if idx > 4:
            break
