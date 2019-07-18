import argparse
import glob
import os
import os.path as osp

from tqdm import tqdm
from skimage.io import imread, imsave


def _list_images(path):
    """Glob all images within a directory."""

    images = []
    for ext in ("jpg", "jpeg", "png", "bmp"):
        images.extend(glob.glob(os.path.join(path, f"*.{ext}")))
    return sorted(images)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('mask_root', help='Path to mask directory')
    parser.add_argument('-o', '--output', help='Path to visualization output')
    args = parser.parse_args()

    mask_root = args.mask_root
    output_dir = args.output
    if output_dir is None:
        output_dir = osp.abspath(osp.join(osp.dirname(mask_root), 'viz'))

    if not osp.exists(output_dir):
        os.mkdir(output_dir)

    for mask_path in tqdm(_list_images(mask_root)):
        imsave(mask_path.replace(mask_root, output_dir),
               imread(mask_path) * 255,
               check_contrast=False)
