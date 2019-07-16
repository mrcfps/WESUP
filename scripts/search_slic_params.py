import argparse
import glob
import os
from itertools import product

import numpy as np
from PIL import Image
from skimage.segmentation import slic
from joblib import Parallel, delayed


def _list_images(path):
    """Glob all images within a directory."""

    images = []
    for ext in ("jpg", "jpeg", "png", "bmp"):
        images.extend(glob.glob(os.path.join(path, f"*.{ext}")))
    return sorted(images)


def read_image(img_path, rescale_factor=0.5, mode=Image.BILINEAR):
    img = Image.open(img_path)
    target_width = int(img.width * rescale_factor)
    target_height = int(img.height * rescale_factor)
    img = img.resize((target_width, target_height), resample=mode)

    return np.array(img)


def run_param_group(img, mask, area, compactness):
    n_segments = int(img.shape[0] * img.shape[1] / area)
    segments = slic(img, n_segments=n_segments, compactness=compactness)
    oracle_pred = np.zeros_like(mask)
    for sp_idx in range(segments.max() + 1):
        sp_mask = segments == sp_idx
        oracle_pred[sp_mask] = mask[sp_mask].mean().round()

    return np.mean(oracle_pred == mask)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_path', help='Path to dataset with images and masks')
    parser.add_argument('-r', '--rescale-factor', default=0.5,
                        help='Rescale factor for resizing images and masks')
    parser.add_argument('-a', '--area', default='50,60,70,80,90,100',
                        help='Approximate number of superpixels')
    parser.add_argument('-c', '--compactness', default='10,20,30,40,50',
                        help='Compactness parameter for SLIC')
    args = parser.parse_args()

    executor = Parallel(os.cpu_count())

    print('Reading images and masks ...')
    image_dir = os.path.join(args.dataset_path, 'images')
    images = executor(delayed(read_image)(img_path)
                      for img_path in _list_images(image_dir))
    mask_dir = os.path.join(args.dataset_path, 'masks')
    masks = executor(delayed(read_image)(mask_path, mode=Image.NEAREST)
                     for mask_path in _list_images(mask_dir))

    areas = [int(n) for n in args.area.split(',')]
    compactnesses = [int(n) for n in args.compactness.split(',')]
    param_groups = product(areas, compactnesses)

    for param_group in param_groups:
        accs = executor(delayed(run_param_group)(img, mask, *param_group)
                        for img, mask in zip(images, masks))
        print(f'# Segments = {param_group[0]}, Compactness = {param_group[1]}, Acc = {np.mean(accs):.4f}')
