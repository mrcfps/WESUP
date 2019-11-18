import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.manifold import TSNE

from models import Wessup
from utils.data import SegmentationDataset

ckpt = torch.load('ckpt.0150.pth')
dataset = SegmentationDataset('LUSC/test', rescale_factor=0.4)

# def compare_tsne(img, mask):
tsne = TSNE()

img, mask = dataset[0]

# before training
print('preparing before training ...')
model = Wessup()
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
