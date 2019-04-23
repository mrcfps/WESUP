"""
Prepare GlaS dataset with the following structure:

1. Training set, in the form of raw RGB images and masks:
    - Images: data/train/images/*.bmp
    - Ground truth: data/train/masks/*.bmp
2. Validation (dev) set, in the form of non-overlapping patches stored in numpy arrays:
    - Images: data/val/images.npy
    - Ground truth: data/val/masks.npy
3. Test set, in the form of whole images and masks in numpy arrays:
    - Images: data/testA/images/*.bmp, data/testB/images/*.bmp
    - Ground truth: data/testA/masks/*.bmp, data/testB/masks/*.bmp
"""

import argparse
import os
import warnings
from shutil import copyfile

import numpy as np
import pandas as pd
from skimage.io import imread, imsave
from sklearn.model_selection import train_test_split

from utils.tile import divide_image_to_patches
import config

warnings.filterwarnings('ignore')

j = os.path.join


def build_cli_parser():
    parser = argparse.ArgumentParser('Dataset generator for GlaS challenge.')
    parser.add_argument(
        'dataset_path', help='Path to original MICCAI 2015 GlaS dataset.')
    parser.add_argument('--val-size', type=float, default=0.1,
                        help='Validation size (between 0 and 1)')
    parser.add_argument('-o', '--output', default='data',
                        help='Path to output dataset')

    return parser


def split_train_val_test(orig_path, val_size=0.1):
    """Split image names into training set and validation set.
    """

    grade = pd.read_csv(j(orig_path, 'Grade.csv'))
    grade.drop(grade.columns[1:3], axis=1, inplace=True)

    testA_set = grade[grade['name'].str.startswith('testA_')]['name']
    testB_set = grade[grade['name'].str.startswith('testB_')]['name']

    grade = grade[grade['name'].str.startswith('train_')]
    grade.columns = ('name', 'grade')
    grade['grade'] = pd.factorize(grade['grade'])[0]

    x, y = grade['name'], grade['grade']
    train_set, val_set, _, _ = train_test_split(
        x, y, test_size=val_size, stratify=y)

    return train_set, val_set, testA_set, testB_set


def prepare_images(orig_path, dst_path, names, split=False):
    if not os.path.exists(dst_path):
        os.mkdir(dst_path)

    dst_img_dir = j(dst_path, 'images')
    dst_mask_dir = j(dst_path, 'masks')
    os.mkdir(dst_img_dir)
    os.mkdir(dst_mask_dir)

    for name in names:
        img_name = '{}.bmp'.format(name)
        mask_name = '{}_anno.bmp'.format(name)

        if split:
            img = imread(os.path.join(orig_path, img_name))
            mask = (imread(os.path.join(orig_path, mask_name)) > 0).astype('uint8')
            img_patches = divide_image_to_patches(img, config.PATCH_SIZE, pad=False)
            mask_patches = divide_image_to_patches(mask, config.PATCH_SIZE, pad=False)

            for idx, (img_patch, mask_patch) in enumerate(zip(img_patches, mask_patches)):
                imsave(j(dst_img_dir, f'{name}-{idx}.bmp'), np.squeeze(img_patch))
                imsave(j(dst_mask_dir, f'{name}-{idx}.bmp'), np.squeeze(mask_patch))
        else:
            orig_img_path = j(orig_path, img_name)
            dst_img_path = j(dst_img_dir, img_name)
            orig_mask_path = j(orig_path, mask_name)
            dst_mask_path = j(dst_mask_dir, img_name)

            # copy original image to destination
            copyfile(orig_img_path, dst_img_path)

            # save binarized mask to destination
            imsave(dst_mask_path, (imread(orig_mask_path) > 0).astype('uint8'))


if __name__ == '__main__':
    parser = build_cli_parser()
    args = parser.parse_args()

    train_set, val_set, testA_set, testB_set = split_train_val_test(
        args.dataset_path, args.val_size)

    if not os.path.exists(args.output):
        os.mkdir(args.output)

    train_dir = j(args.output, 'train')
    val_dir = j(args.output, 'val')
    testA_dir = j(args.output, 'testA')
    testB_dir = j(args.output, 'testB')

    prepare_images(args.dataset_path, train_dir, train_set)
    print('Training data is done.')

    prepare_images(args.dataset_path, val_dir, val_set, split=True)
    print('Validation data is done.')

    prepare_images(args.dataset_path, testA_dir, testA_set)
    print('TestA data is done.')

    prepare_images(args.dataset_path, testB_dir, testB_set)
    print('TestB data is done.')
