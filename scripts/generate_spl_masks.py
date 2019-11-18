import csv
import os
from pathlib import Path

import fire
import numpy as np
from joblib import Parallel, delayed
from skimage.io import imread
from skimage.segmentation import slic


def generate(data_root, n_classes=2, sp_area=200, compactness=40):
    # TODO: add expanduser to all similar statements
    data_root = Path(data_root).expanduser()
    img_dir = data_root / 'images'

    def generate_spl_mask(img_path, point_path):
        img = imread(img_path)
        height, width = img.shape[:2]

        with open(point_path) as fp:
            points = np.array([[int(d) for d in point]
                               for point in csv.reader(fp)])

        n_segments = (height * width) // sp_area
        segments = slic(img, n_segments=n_segments, compactness=compactness)

        mask = np.zeros((height, width, n_classes), dtype='uint8')
        for point in points:
            y, x, class_ = point
            mask[segments == segments[x, y], class_] = 1

        return mask

    executor = Parallel(os.cpu_count())

    for point_dir in data_root.glob('points*'):
        if not point_dir.is_dir():
            continue

        print(f'Processing {point_dir} ...')
        img_paths = sorted(img_dir.iterdir())
        point_paths = sorted(point_dir.iterdir())

        spl_masks = executor(delayed(generate_spl_mask)(img_path, point_path)
                             for img_path, point_path in zip(img_paths, point_paths))

        output_dir = data_root / point_dir.name.replace('points', 'spl-masks')
        if not output_dir.exists():
            output_dir.mkdir()

        for img_path, spl_mask in zip(img_paths, spl_masks):
            np.save(output_dir / img_path.name.replace(img_path.suffix, '.npy'), spl_mask)

        print(f'Saved to {output_dir}.')


if __name__ == '__main__':
    fire.Fire(generate)
