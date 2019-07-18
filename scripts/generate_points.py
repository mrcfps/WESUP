"""
Script for generating point annotation.
"""

import argparse
import csv
import os
import sys

import numpy as np
from tqdm import tqdm
from PIL import Image
from joblib import Parallel, delayed


def _sample_within_region(region_mask, class_label, num_samples=1):
    xs, ys = np.where(region_mask)
    selected_indexes = np.random.permutation(len(xs))[:num_samples]
    xs, ys = xs[selected_indexes], ys[selected_indexes]

    return np.c_[xs, ys, np.full_like(xs, class_label)]


def _generate_points(mask, point_ratio=1e-4):
    points = []

    # loop over all class labels
    # (from 0 to n_classes, where 0 is background)
    for class_label in np.unique(mask):
        class_mask = mask == class_label
        points.append(
            _sample_within_region(
                class_mask, class_label,
                num_samples=max(int(class_mask.sum() * point_ratio), 1)
            )
        )

    return np.concatenate(points)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Dot annotation generator.')
    parser.add_argument('root_dir',
                        help='Path to data root directory with mask-level annotation.')
    parser.add_argument('-p', '--point-ratio', type=float, default=1e-4,
                        help='Percentage of labeled objects (regions) for each class')
    args = parser.parse_args()

    mask_dir = os.path.join(args.root_dir, 'masks')
    if not os.path.exists(mask_dir):
        print('Cannot generate dot annotation without masks.')
        sys.exit(1)

    label_dir = os.path.join(args.root_dir, f'points-{str(args.point_ratio)}')

    if not os.path.exists(label_dir):
        os.mkdir(label_dir)

    print('Generating point annotation ...')

    def para_func(fname):
        basename = os.path.splitext(fname)[0]
        mask = np.array(Image.open(os.path.join(mask_dir, fname)))
        points = _generate_points(mask, point_ratio=args.point_ratio)

        # conform to the xy format
        points[:, [0, 1]] = points[:, [1, 0]]

        with open(os.path.join(label_dir, f'{basename}.csv'), 'w') as fp:
            writer = csv.writer(fp)
            writer.writerows(points)

    Parallel(n_jobs=os.cpu_count())(delayed(para_func)(fname) for fname in tqdm(os.listdir(mask_dir)))
