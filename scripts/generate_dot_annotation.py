import argparse
import csv
import os
import random
import sys
from shutil import copytree

import numpy as np
from tqdm import tqdm
from PIL import Image
from skimage.measure import label

BG_SAMPLE_RATIO = 5e-5


def _sample_within_region(region_mask, random_sample=False):
    xs, ys = np.where(region_mask)
    x_center, y_center = int(xs.mean().round()), int(ys.mean().round())

    # if the center point is inside the region, return it
    if region_mask[x_center, y_center] and not random_sample:
        return x_center, y_center

    # else return a random point
    return random.choice(np.c_[xs, ys])


def _generate_points(mask, label_percent=0.5):
    points = []

    # loop over all class labels
    # (from 0 to n_classes, where 0 is background)
    for class_label in np.unique(mask):
        class_mask = mask == class_label
        if class_label == 0:
            # if background, randomly sample some points
            sample_num = int(class_mask.sum() * BG_SAMPLE_RATIO)
            for _ in range(sample_num):
                point = _sample_within_region(class_mask, random_sample=True)
                points.append([*point, class_label])
        else:
            class_mask = label(class_mask)

            # randomly select part of regions
            region_indexes = np.unique(class_mask)
            region_indexes = region_indexes[region_indexes > 0]
            np.random.shuffle(region_indexes)
            num_selected = int(np.ceil(len(region_indexes) * label_percent))
            region_indexes = region_indexes[:num_selected]

            for idx in region_indexes:
                point = _sample_within_region(class_mask == idx)
                points.append([*point, class_label])

    return points


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Dot annotation generator.')
    parser.add_argument('root_dir', help='Path to data root directory with mask-level annotation.')
    parser.add_argument('-p', '--label-percent', type=float, default=0.5,
                        help='Percentage of labeled objects (regions) for each class')
    args = parser.parse_args()

    mask_dir = os.path.join(args.root_dir, 'masks')
    if not os.path.exists(mask_dir):
        print('Cannot generate dot annotation without masks.')
        sys.exit(1)

    label_dir = os.path.join(args.root_dir, 'labels')

    if not os.path.exists(label_dir):
        os.mkdir(label_dir)

    print('Generate labels with dot annotation ...')
    for fname in tqdm(os.listdir(mask_dir)):
        basename = os.path.splitext(fname)[0]
        mask = np.array(Image.open(os.path.join(mask_dir, fname)))
        points = _generate_points(mask, label_percent=args.label_percent)

        with open(os.path.join(label_dir, f'{basename}.csv'), 'w') as fp:
            writer = csv.writer(fp)
            writer.writerows(points)
