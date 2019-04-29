import argparse
import csv
import os

import cv2
from tqdm import tqdm
from skimage.io import imread, imsave

COLORS = (
    (0, 0, 255),
    (255, 0, 0),
    (0, 255, 0),
)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_path', help='Path to dot annotation dataset')
    parser.add_argument('-r', '--radius', type=int, default=5, help='Circle radius')
    parser.add_argument('-o', '--output',
                        help='Output path to store visualization results')
    args = parser.parse_args()

    output_dir = args.output or os.path.join(args.dataset_path, 'viz')

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    img_dir = os.path.join(args.dataset_path, 'images')
    label_dir = os.path.join(args.dataset_path, 'labels')

    print(f'Generating dot annotation visualizaion to {output_dir} ...')
    for img_name in tqdm(os.listdir(img_dir)):
        basename = os.path.splitext(img_name)[0]
        img = imread(os.path.join(img_dir, img_name))
        csvfile = open(os.path.join(label_dir, f'{basename}.csv'))
        csvreader = csv.reader(csvfile)

        for point in csvreader:
            point = [int(d) for d in point]
            center = (point[0])
            cv2.circle(img, (point[1], point[0]), args.radius, COLORS[point[2]], -1)

        imsave(os.path.join(output_dir, img_name), img)
        csvfile.close()
