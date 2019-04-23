import argparse
import csv
import os
import random
from shutil import copytree

import numpy as np
from tqdm import tqdm
from PIL import Image
from skimage.measure import label

# shorthand for joining paths
j = os.path.join

BG_SAMPLE_RATIO = 1e-4


def _sample_within_region(region_mask):
    xs, ys = np.where(region_mask)
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
                point = _sample_within_region(class_mask)
                points.append([*point, class_label])
        else:
            class_mask = label(class_mask)

            # randomly select part of regions
            region_indexes = np.unique(class_mask)
            np.random.shuffle(region_indexes)
            num_selected = int(np.ceil(len(region_indexes) * label_percent))
            region_indexes = region_indexes[:num_selected]

            for idx in region_indexes:
                point = _sample_within_region(class_mask == idx)
                points.append([*point, class_label])

    return points


def _write_points_to_csv(points, file_path):
    with open(j(dst_label_dir, f'{basename}.csv'), 'w') as fp:
        writer = csv.writer(fp)
        writer.writerows(points)


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

    print('Copy all training images ...')
    copytree(j(src_train_dir, 'images'), j(dst_train_dir, 'images'))

    print('Generate labels with dot annotation ...')
    src_mask_dir = j(src_train_dir, 'masks')
    for fname in tqdm(os.listdir(src_mask_dir)):
        basename = os.path.splitext(fname)[0]
        mask = np.array(Image.open(j(src_mask_dir, fname)))
        points = _generate_points(mask, label_percent=args.label_percent)

        with open(j(dst_label_dir, f'{basename}.csv'), 'w') as fp:
            writer = csv.writer(fp)
            writer.writerows(points)

    print('Copy validation data ...')
    copytree(j(args.dataset_path, 'val'), j(args.output, 'val'))
