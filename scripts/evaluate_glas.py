import argparse
import glob
import os
import os.path as osp

import numpy as np
import pandas as pd
from skimage.io import imread, imsave
from joblib import Parallel, delayed

from utils.metrics import *

parser = argparse.ArgumentParser()
parser.add_argument('pred_root')
args = parser.parse_args()

pred_root = args.pred_root
new_pred_root = pred_root + '-new'
if not osp.exists(new_pred_root):
    os.mkdir(new_pred_root)
    os.mkdir(osp.join(new_pred_root, 'testA'))
    os.mkdir(osp.join(new_pred_root, 'testB'))

executor = Parallel(n_jobs=os.cpu_count())


def postprocess(pred):
    regions = label(pred)
    for region_idx in range(regions.max() + 1):
        region_mask = regions == region_idx
        if region_mask.sum() < 1000:
            pred[region_mask] = 0

    revert_regions = label(1 - pred)
    for region_idx in range(revert_regions.max() + 1):
        region_mask = revert_regions == region_idx
        if region_mask.sum() < 1000:
            pred[region_mask] = 1

    return pred


def compute_metrics(predictions, gts, pred_paths):
    iterable = list(zip(predictions, gts))

    accuracies = executor(delayed(accuracy)(pred, gt) for pred, gt in iterable)
    print('Accuracy:', np.mean(accuracies))

    dices = executor(delayed(dice)(pred, gt) for pred, gt in iterable)
    print('Dice:', np.mean(dices))

    detection_f1s = executor(delayed(detection_f1)(pred, gt) for pred, gt in iterable)
    print('Detection F1:', np.mean(detection_f1s))

    object_dices = executor(delayed(object_dice)(pred, gt) for pred, gt in iterable)
    print('Object Dice:', np.mean(object_dices))

    object_hausdorffs = executor(delayed(object_hausdorff)(pred, gt) for pred, gt in iterable)
    print('Object Hausdorff:', np.mean(object_hausdorffs))

    df = pd.DataFrame()
    df['detection_f1'] = detection_f1s
    df['object_dice'] = object_dices
    df['object_hausdorff'] = object_hausdorffs
    df.index = [osp.basename(pred_path) for pred_path in pred_paths]

    return df


print('Test A')

print('\nReading predictions and gts ...')
pred_paths = sorted(glob.glob(osp.join(pred_root, 'testA', '*.bmp')))
predictions = executor(delayed(postprocess)(imread(pred_path) / 255) for pred_path in pred_paths)
gts = executor(delayed(imread)(gt_path) for gt_path in sorted(glob.glob('data_glas_all/testA/masks/*.bmp')))

print('Saving new predictions ...')
for pred, pred_path in zip(predictions, pred_paths):
    imsave(pred_path.replace(pred_root, pred_root + '-new'), (pred * 255).astype('uint8'))

metrics = compute_metrics(predictions, gts, pred_paths)
metrics.to_csv(osp.join(pred_root, 'testA.csv'))

print('\nTest B')

print('\nReading predictions and gts ...')
pred_paths = sorted(glob.glob(osp.join(pred_root, 'testB', '*.bmp')))
predictions = executor(delayed(postprocess)(imread(pred_path) / 255) for pred_path in pred_paths)
gts = executor(delayed(imread)(gt_path) for gt_path in sorted(glob.glob('data_glas_all/testB/masks/*.bmp')))

print('Saving new predictions ...')
for pred, pred_path in zip(predictions, pred_paths):
    imsave(pred_path.replace(pred_root, pred_root + '-new'), (pred * 255).astype('uint8'))

metrics = compute_metrics(predictions, gts, pred_paths)
metrics.to_csv(osp.join(pred_root, 'testB.csv'))