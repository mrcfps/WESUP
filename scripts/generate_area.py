"""
Script for generating area information, used in CWDS-MIL.
"""

import argparse
import sys
import os

import pandas as pd
from skimage.io import imread


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Area information generator.')
    parser.add_argument('root_dir',
                        help='Path to data root directory with mask-level annotation.')
    args = parser.parse_args()

    mask_dir = os.path.join(args.root_dir, 'masks')
    if not os.path.exists(mask_dir):
        print('Cannot generate area information without masks.')
        sys.exit(1)

    area_info = pd.DataFrame(columns=['img', 'area'])

    for idx, img_name in enumerate(sorted(os.listdir(mask_dir))):
        img_path = os.path.join(mask_dir, img_name)
        img = imread(img_path)
        area_info.loc[idx] = [img_name, img.mean()]

    output_path = os.path.join(args.root_dir, 'area.csv')
    area_info.to_csv(output_path)
    print(f'Area information saved to {output_path}.')
