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
from skimage.measure import label


def _sample_within_region(region_mask, class_label, num_samples=1):

    xs, ys = np.where(region_mask)

    if num_samples == 1:
        x_center, y_center = int(xs.mean().round()), int(ys.mean().round())

        retry = 0
        while True:
            # deviate from the center within a circle within radius 5
            x = x_center + np.random.randint(-5, 6)
            y = y_center + np.random.randint(-5, 6)

            # if the center point is inside the region, return it
            try:
                if region_mask[x, y]:
                    return np.c_[x, y, class_label]
            except IndexError:
                pass
            finally:
                retry += 1

            if retry > 5:
                break

    selected_indexes = np.random.permutation(len(xs))[:num_samples]
    xs, ys = xs[selected_indexes], ys[selected_indexes]

    return np.c_[xs, ys, np.full_like(xs, class_label)]


def _generate_points(mask, point_ratio=1e-4):
    points = []

    # loop over all class labels
    # (from 0 to n_classes, where 0 is background)
    for class_label in np.unique(mask):
        class_mask = mask == class_label
        if class_label == 0:
            # if background, randomly sample some points
            points.append(
                _sample_within_region(
                    class_mask, class_label,
                    num_samples=int(class_mask.sum() * point_ratio)
                )
            )
        else:
            class_mask = label(class_mask)
            region_indexes = np.unique(class_mask)
            region_indexes = region_indexes[np.nonzero(region_indexes)]

            # iterate over all instances of this class
            for idx in np.unique(region_indexes):
                region_mask = class_mask == idx
                num_samples = max(1, int(region_mask.sum() * point_ratio))
                points.append(
                    _sample_within_region(
                        region_mask, class_label, num_samples=num_samples
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
    for fname in tqdm(os.listdir(mask_dir)):
        basename = os.path.splitext(fname)[0]
        mask = np.array(Image.open(os.path.join(mask_dir, fname)))
        points = _generate_points(mask, point_ratio=args.point_ratio)

        # conform to the xy format
        points[:, [0, 1]] = points[:, [1, 0]]

        with open(os.path.join(label_dir, f'{basename}.csv'), 'w') as fp:
            writer = csv.writer(fp)
            writer.writerows(points)
