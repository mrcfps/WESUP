import argparse
import os
import glob
from pathlib import Path

import numpy as np
from tqdm import tqdm
from skimage.io import imread, imsave
from joblib import Parallel, delayed

parser = argparse.ArgumentParser()
parser.add_argument('dataset_path', help='path to dataset')
parser.add_argument('-o', '--output', required=True,
                    help='where to store target dataset')
parser.add_argument('-p', '--patch-size', type=int, default=464,
                    help='patch size of target dataset')
args = parser.parse_args()

patch_size = args.patch_size

train_dir = Path(args.dataset_path)
train_dir.mkdir(exist_ok=True)

img_dir = train_dir / 'images'
mask_dir = train_dir / 'masks'
img_dir.mkdir(exist_ok=True)
mask_dir.mkdir(exist_ok=True)


def process_img_and_mask(img_path, mask_path, idx, n_patches=12):
    img = imread(img_path)
    mask = imread(mask_path)
    h, w = img.shape[:2]
    basename = os.path.basename(img_path)

    for n in range(n_patches):
        rand_i = int(np.random.randint(0, h - patch_size))
        rand_j = int(np.random.randint(0, w - patch_size))
        img_patch = img[rand_i:rand_i + patch_size, rand_j:rand_j + patch_size]
        mask_patch = mask[rand_i:rand_i + patch_size, rand_j:rand_j + patch_size]
        imsave(str(img_dir / basename.replace('.png', f'_{n}.png')), img_patch, check_contrast=False)
        imsave(str(mask_dir / basename.replace('.png', f'_{n}.png')), mask_patch, check_contrast=False)


executor = Parallel(n_jobs=12)

img_paths = sorted(glob.glob(f'{args.output}/images/*.png'))
mask_paths = sorted(glob.glob(f'{{args.output}}/masks/*.png'))

# images = executor(delayed(imread)(img_path) for img_path in tqdm(img_paths))
# masks = executor(delayed(imread)(mask_path) for mask_path in tqdm(mask_paths))
# for img_path in img_paths:
#     print('reading', img_path)
#     imread(img_path)

print('\nSplitting into patches ...')
executor(delayed(process_img_and_mask)(img_path, mask_path, idx)
         for idx, (img_path, mask_path) in tqdm(enumerate(zip(img_paths, mask_paths)),
                                                total=len(img_paths)))
