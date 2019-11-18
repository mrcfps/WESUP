import argparse
from pathlib import Path
from skimage.io import imread
from utils.metrics import *

GT_PATH = Path('~/data/GLAS_all/train/masks').expanduser()

parser = argparse.ArgumentParser()
parser.add_argument('test_dir')
args = parser.parse_args()

test_dir = Path(args.test_dir).expanduser()

accs

for img_name in test_dir.glob('*.tif'):
    pred = imread(str(img_name))
    pred = pred.sum(axis=-1)
    pred[pred == 65280] = 1
    pred[pred == 118784] = 0

