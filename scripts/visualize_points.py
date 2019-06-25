"""
Script for visualizing point annotation.
"""

import argparse
import csv
import os
import os.path as osp

import cv2
from tqdm import tqdm
from skimage.io import imread, imsave

COLORS = (
    (0, 255, 0),
    (255, 0, 0),
)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('point_root', help='Path to point labels directory')
    parser.add_argument('-r', '--radius', type=int, default=5, help='Circle radius')
    parser.add_argument('-o', '--output',
                        help='Output path to store visualization results')
    args = parser.parse_args()

    output_dir = args.output or osp.join(args.point_root, 'viz')

    if not osp.exists(output_dir):
        os.mkdir(output_dir)

    img_dir = osp.join(osp.dirname(args.point_root), 'images')

    print(f'Generating dot annotation visualizaion to {output_dir} ...')
    for img_name in tqdm(os.listdir(img_dir)):
        basename = osp.splitext(img_name)[0]
        img = imread(osp.join(img_dir, img_name))
        csvfile = open(osp.join(args.point_root, f'{basename}.csv'))
        csvreader = csv.reader(csvfile)

        for point in csvreader:
            point = [int(d) for d in point]
            cv2.circle(img, (point[0], point[1]), args.radius, COLORS[point[2]], -1)

        imsave(osp.join(output_dir, img_name), img)
        csvfile.close()
