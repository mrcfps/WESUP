import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.manifold import TSNE

from models import WESUP
from utils.data import SegmentationDataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', help='Path to trained model')
    parser.add_argument('dataset_path', help='Path to test dataset')
    parser.add_argument('--index', type=int, default=0, help='Sample index')
    parser.add_argument('--rescale-factor', type=float, default=0.4, help='Rescale factor for input samples')
    args = parser.parse_args()

    ckpt = torch.load(args.model_path)
    dataset = SegmentationDataset(args.dataset_path, rescale_factor=args.rescale_factor)

    tsne = TSNE()
    img, mask = dataset[args.index]

    # before training
    print('preparing before training ...')
    model = WESUP()
    (img, sp_maps), (pixel_mask, sp_labels) = model.preprocess(img, mask)
    sp_labels = sp_labels.argmax(dim=1).numpy()
    pred = model((img.unsqueeze(0), sp_maps))
    before_features = model._sp_features.detach().numpy()
    before_x2d = tsne.fit_transform(before_features)

    # after training
    print('preparing after training ...')
    model.load_state_dict(ckpt['model_state_dict'])
    pred = model((img.unsqueeze(0), sp_maps))
    after_features = model._sp_features.detach().numpy()
    after_x2d = tsne.fit_transform(after_features)

    # plotting
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.scatter(before_x2d[:, 0], before_x2d[:, 1], c=sp_labels, alpha=0.3)
    ax2.scatter(after_x2d[:, 0], after_x2d[:, 1], c=sp_labels, alpha=0.3)
    plt.show()

    np.savez('tsne-crag.npz', before_x2d=before_x2d, after_x2d=after_x2d, sp_labels=sp_labels)
