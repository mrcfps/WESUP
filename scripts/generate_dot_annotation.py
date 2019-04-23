"""
Utility script for generating dataset with dot annotation.
"""

import argparse
import csv
import os
from shutil import copytree

import numpy as np
from PIL import Image
from skimage.measure import label

# shorthand for joining paths
j = os.path.join

BG_SAMPLE_RATIO = 1e-4


def _sample_within_region(region_mask):
    while True:
        # get the bounds of this region
        up = max(0, region_mask.any(axis=1).nonzero()[0][0])
        bottom = min(region_mask.shape[0] - 1,
                     region_mask.any(axis=1).nonzero()[0][-1])
        left = max(0, region_mask.any(axis=0).nonzero()[0][0])
        right = min(region_mask.shape[1] - 1,
                    region_mask.any(axis=0).nonzero()[0][-1])

        rand_i = np.random.randint(up, bottom + 1)
        rand_j = np.random.randint(left, right + 1)

        if region_mask[rand_i, rand_j]:
            # point within the region is found, return it
            return rand_i, rand_j


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
                points.append(
                    [*_sample_within_region(class_mask), class_label])
        else:
            class_mask = label(class_mask)

            # randomly select part of regions
            region_indexes = np.unique(class_mask)
            np.random.shuffle(region_indexes)
            num_selected = int(np.ceil(len(region_indexes) * label_percent))
            region_indexes = region_indexes[:num_selected]

            points.extend([
                [*_sample_within_region(class_mask == idx), class_label]
                for idx in region_indexes
            ])

    return points


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Dot annotation generator.')
    parser.add_argument('dataset_path',
                        help='Path to source dataset with mask-level annotation.')
    parser.add_argument('-p', '--label-percent', type=float, default=0.5,
                        help='Percentage of labeled objects (regions) for each class')
    parser.add_argument('-o', '--output', default='dot_anno_data',
                        help='Output directory for dataset with dot annotation')
    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.mkdir(args.output)

    src_train_dir = j(args.dataset_path, 'train')
    dst_train_dir = j(args.output, 'train')
    dst_label_dir = j(dst_train_dir, 'labels')
    if not os.path.exists(dst_train_dir):
        os.mkdir(dst_train_dir)
        os.mkdir(dst_label_dir)

    # copy all training images
    copytree(j(src_train_dir, 'images'), j(dst_train_dir, 'images'))

    # generate labels with dot annotation
    src_mask_dir = j(src_train_dir, 'masks')
    for fname in os.listdir(src_mask_dir):
        basename = os.path.splitext(fname)[0]
        mask = np.array(Image.open(j(src_mask_dir, fname)))
        points = _generate_points(mask, label_percent=args.label_percent)

        with open(j(dst_label_dir, f'{basename}.csv'), 'w') as fp:
            writer = csv.writer(fp)
            writer.writerows(points)

    # copy the entire validation set
    copytree(j(args.dataset_path, 'val'), j(args.output, 'val'))
