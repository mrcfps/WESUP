import argparse
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

train_dir = Path(args.dataset_path).expanduser()
train_dir.mkdir(exist_ok=True)

img_dir = train_dir / 'images'
mask_dir = train_dir / 'masks'
img_dir.mkdir(exist_ok=True)
mask_dir.mkdir(exist_ok=True)

output_dir = Path(args.output).expanduser()
output_dir.mkdir(exist_ok=True)
target_img_dir = output_dir / 'images'
target_mask_dir = output_dir / 'masks'
target_img_dir.mkdir(exist_ok=True)
target_mask_dir.mkdir(exist_ok=True)


def process_img_and_mask(img_path, mask_path, n_patches=12):
    img = imread(img_path)
    mask = imread(mask_path)
    h, w = img.shape[:2]
    name = img_path.name
    suffix = img_path.suffix

    for n in range(n_patches):
        rand_i = int(np.random.randint(0, h - patch_size))
        rand_j = int(np.random.randint(0, w - patch_size))
        img_patch = img[rand_i:rand_i + patch_size, rand_j:rand_j + patch_size]
        mask_patch = mask[rand_i:rand_i + patch_size, rand_j:rand_j + patch_size]
        mask_patch = (mask_patch / 255).astype('uint8')
        patch_name = name.replace(suffix, f'_{n}{suffix}')
        imsave(str(target_img_dir / patch_name), img_patch, check_contrast=False)
        imsave(str(target_mask_dir / patch_name), mask_patch, check_contrast=False)


executor = Parallel(n_jobs=12)

img_paths = sorted(img_dir.iterdir())
mask_paths = sorted(mask_dir.iterdir())

print('\nSplitting into patches ...')
executor(delayed(process_img_and_mask)(img_path, mask_path)
         for img_path, mask_path in tqdm(zip(img_paths, mask_paths), total=len(img_paths)))
# for img_path, mask_path in tqdm(zip(img_paths, mask_paths), total=len(img_paths)):
#     process_img_and_mask(img_path, mask_path)
