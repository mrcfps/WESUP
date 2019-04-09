import argparse
import os
import time

import numpy as np
from PIL import Image
from skimage.segmentation import slic, mark_boundaries

import torch

import config
from extractors import VGG16FeatureExtractor

device = torch.device(config.DEVICE)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('train_dir', help='Path to training data directory.')
    args = parser.parse_args()

    # all training code (e.g., train_1, train_7, etc)
    images = set(fname.replace('_anno', '')
                 for fname in os.listdir(args.train_dir)
                 if fname.endswith('bmp'))
    print(f'Total of {len(images)} training images detected.')

    x, y = None, None

    for img_name in images:
        print(f'Processing {img_name} ...')

        anno_name = img_name.replace('.bmp', '_anno.bmp')
        img = Image.open(os.path.join(args.train_dir, img_name))
        anno = Image.open(os.path.join(args.train_dir, anno_name))

        try:
            start = time.time()
            extractor = VGG16FeatureExtractor()
            new_x, new_y = extractor.extract(img, anno)
            extractor.close()
            end = time.time()

            x = torch.cat((x, new_x)) if x is not None else new_x
            y = torch.cat((y, new_y)) if y is not None else new_y

            print('Processed {}, took {:.3f} seconds.'.format(
                img_name, end - start))
        except Exception as e:
            print(f'{img_name} failed')
            print(e)
            torch.save(x, 'features.pth')
            torch.save(y, 'labels.pth')

    print('x.size()', x.size())
    print('y.size()', y.size())
    torch.save(x, 'features.pth')
    torch.save(y, 'labels.pth')
