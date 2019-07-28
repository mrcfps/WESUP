import argparse
import glob
import os
import os.path as osp
from pathlib import Path
from itertools import product

import numpy as np
from tqdm import tqdm
from skimage.io import imread, imsave
from skimage.measure import label
from joblib import Parallel, delayed

colors = list(product([0, 64, 128, 192, 255],
                      [0, 64, 128, 192, 255],
                      [0, 64, 128, 192, 255]))

# filter out colors close to background
colors = [color for color in colors if 192 < sum(color) < 765]

np.random.seed(42)
np.random.shuffle(colors)


def _list_images(path):
    """Glob all images within a directory."""

    images = []
    for ext in ("jpg", "jpeg", "png", "bmp"):
        images.extend(path.glob(f"*.{ext}"))
    return sorted(images)


def iou(a, b):
    return (a & b).sum() / (a | b).sum()


def paint(mask):
    painted = np.zeros((*mask.shape, 3), dtype='uint8')

    for region_id in np.unique(mask):
        if region_id >= len(colors):
            painted[mask == region_id] = np.random.randint(0, 256, size=(3,), dtype='uint8')
        if 0 < region_id < len(colors):
            painted[mask == region_id] = colors[region_id]

    return painted


def paint_pred_and_gt(pred, gt):
    pred, gt = label(pred), label(gt)
    new_pred = np.zeros_like(pred)
    max_id = max(pred.max(), gt.max())

    for pred_region_id in range(1, pred.max() + 1):
        pred_region = pred == pred_region_id
        matched_gts = []

        for gt_region_id in range(1, gt.max() + 1):
            gt_region = gt == gt_region_id
            intersect = (pred_region & gt_region).sum() / gt_region.sum()

            if intersect > 0.5:
                matched_gts.append((gt_region, gt_region_id))

        if len(matched_gts) > 0:
            # choose the largest matched ground truth object
            new_pred[pred_region] = max(matched_gts, key=lambda x: x[0].sum())[1]
        else:
            new_pred[pred_region] = max_id + pred_region_id

    return paint(new_pred), paint(gt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('pred_path', help='Path to model predictions')
    parser.add_argument('gt_path', help='Path to ground truth masks')
    parser.add_argument('-m', '--model', help='Model name')
    parser.add_argument('-o', '--output', help='Path to output directory')
    args = parser.parse_args()

    pred_path = Path(args.pred_path)
    gt_path = Path(args.gt_path)
    executor = Parallel(os.cpu_count())

    pred_masks = _list_images(pred_path)
    gt_masks = _list_images(gt_path)
    n_examples = len(pred_masks)

    print('Reading predictions and masks ...')
    pred_list = executor(delayed(imread)(mask_path) for mask_path in pred_masks)
    gt_list = executor(delayed(imread)(mask_path) for mask_path in gt_masks)

    print('Painting beautiful illustrations ...')
    paintings = executor(delayed(paint_pred_and_gt)(pred, gt)
                         for pred, gt in tqdm(zip(pred_list, gt_list), total=n_examples))

    if args.output is None:
        output_dir = pred_path.parent / 'paintings'
    else:
        output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)

    print(f'Saving paintings to {output_dir} ...')
    for (pred, gt), pred_path in tqdm(zip(paintings, pred_masks), total=n_examples):
        pred_name = f'{pred_path.stem}.{args.model or "pred"}.png'
        gt_name = f'{pred_path.stem}.gt.png'
        imsave(str(output_dir / pred_name), pred, check_contrast=False)
        imsave(str(output_dir / gt_name), gt, check_contrast=False)

    print('Done')
