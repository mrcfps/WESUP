import argparse
import glob
import os
import os.path as osp
from pathlib import Path
from tqdm import tqdm

import numpy as np
from skimage.io import imread, imsave
from skimage.measure import label
from skimage.morphology import closing
from joblib import Parallel, delayed

from utils.metrics import *


def read_image(img_path):
    img = imread(img_path)
    img = img[..., :3].sum(axis=-1)
    img[img == 464] = 0
    img[img == 255] = 1
    img = closing(img)
    return img.astype('uint8')


def postprocess(img, threshold=5000):
    regions = label(img)
    result = img.copy()
    for region_idx in range(regions.max() + 1):
        region_mask = regions == region_idx
        if region_mask.sum() < threshold:
            result[region_mask] = 0

    revert_regions = label(1 - result)
    for region_idx in range(revert_regions.max() + 1):
        region_mask = revert_regions == region_idx
        if region_mask.sum() < threshold:
            result[region_mask] = 1

    return result


def compute_metrics(iterable):
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


if __name__ == '__main__':
    test_root = '/home/mrc/data/CRAG/test'
    wildcard = '*.png'

    parser = argparse.ArgumentParser()
    parser.add_argument('pred_root')
    args = parser.parse_args()

    pred_root = args.pred_root
    new_pred_root = pred_root + '-new'
    if not osp.exists(new_pred_root):
        os.mkdir(new_pred_root)

    executor = Parallel(n_jobs=os.cpu_count())

    print('Reading predictions  ...')
    pred_paths = sorted(glob.glob(osp.join(pred_root, wildcard)))
    predictions = executor(delayed(postprocess)(read_image(pred_path)) for pred_path in tqdm(pred_paths))
    # predictions = executor(delayed(read_image)(pred_path) for pred_path in pred_paths)
    print('Reading gts ...')
    gts = executor(delayed(imread)(gt_path) for gt_path in tqdm(sorted(glob.glob(osp.join(test_root, 'masks', wildcard)))))

    print('Saving new predictions ...')
    for pred, pred_path in zip(predictions, pred_paths):
        imsave(pred_path.replace(pred_root, pred_root + '-new'), (pred * 255).astype('uint8'))

    compute_metrics(list(zip(predictions, gts)))
