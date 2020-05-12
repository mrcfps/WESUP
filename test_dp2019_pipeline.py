import argparse
import csv
import math
import cv2
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
from skimage.measure import label
from joblib import Parallel, delayed
from itertools import product
from shutil import rmtree

from infer import main as infer
from pixel_infer import main as pixel_infer


def split_patches(data_root, patch_size):
    img_dir = data_root / 'images'
    mask_dir = data_root / 'masks'

    img_paths = sorted(list(img_dir.glob('*.jpg')))
    mask_paths = sorted(list(mask_dir.glob('*.png')))

    output_dir = data_root.parent / f'{data_root.name}-patches'

    if output_dir.exists():
        print(f'{output_dir} found. Skipping.')
        return output_dir

    output_dir.mkdir(exist_ok=True)
    target_img_dir = output_dir / 'images'
    target_mask_dir = output_dir / 'masks'
    target_img_dir.mkdir(exist_ok=True)
    target_mask_dir.mkdir(exist_ok=True)

    def split(img, mask, index):
        height, width, channels = img.shape
        ext_height = math.ceil(height / patch_size) * patch_size
        ext_width = math.ceil(width / patch_size) * patch_size

        ext_img = np.zeros((ext_height, ext_width, channels), dtype=img.dtype)
        ext_mask = np.zeros((ext_height, ext_width), dtype=mask.dtype)
        ext_img[:height, :width] = img
        ext_mask[:height, :width] = mask

        xys = list(product(range(0, width + 1, patch_size),
                           range(0, height + 1, patch_size)))
        for x, y in xys:
            img_patch = ext_img[y:y + patch_size, x:x + patch_size]
            dest = str(target_img_dir / f'{index}-{x}-{y}.jpg')
            cv2.imwrite(dest, img_patch)

            mask_patch = ext_mask[y:y + patch_size, x:x + patch_size]
            dest = str(target_mask_dir / f'{index}-{x}-{y}.png')
            cv2.imwrite(dest, mask_patch)

        return height, width

    info = []

    for index, (img_path, mask_path) in tqdm(enumerate(zip(img_paths, mask_paths)), total=len(img_paths)):
        img = cv2.imread(str(img_path))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        info.append((img_path.stem, *split(img, mask, index)))

    with open(output_dir / 'info.csv', 'w') as fp:
        writer = csv.writer(fp)
        writer.writerows(info)

    return output_dir


def accuracy(P, G):
    return (P == G).mean()


def dice(S, G, epsilon=1e-7):
    S, G = S > 0, G > 0
    dice_score = 2 * (G * S).sum() / (G.sum() + S.sum() + epsilon)
    return dice_score


def postprocess(pred, threshold=1000):
    regions = label(pred)
    for region_idx in range(regions.max() + 1):
        region_mask = regions == region_idx
        if region_mask.sum() < threshold:
            pred[region_mask] = 0

    revert_regions = label(255 - pred)
    for region_idx in range(revert_regions.max() + 1):
        region_mask = revert_regions == region_idx
        if region_mask.sum() < threshold:
            pred[region_mask] = 255

    return pred


def compute_metrics(predictions, gts, pred_paths, negative=False):
    if negative:
        predictions = [(255 - pred) for pred in predictions]
        gts = [(255 - gt) for gt in gts]

    iterable = list(zip(predictions, gts))

    accuracies = executor(delayed(accuracy)(pred, gt) for pred, gt in iterable)
    print('Accuracy:', np.mean(accuracies))

    dices = executor(delayed(dice)(pred, gt) for pred, gt in iterable)
    print('Dice:', np.mean(dices))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('data_root')
    parser.add_argument(
        '-m', '--model', choices=['fcn', 'cdws', 'wesup', 'sizeloss'], default='wesup')
    parser.add_argument('--pixel', action='store_true', default=False)
    parser.add_argument('--skip-infer', action='store_true', default=False)
    parser.add_argument('-p', '--patch-size', type=int, default=1000)
    parser.add_argument('-c', '--checkpoint', help='Path to checkpoint')
    parser.add_argument('--device', help='Device to use')

    args = parser.parse_args()
    print(args)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    data_root = Path(args.data_root).expanduser()

    print('\nSplitting patches ...')
    patch_dir = split_patches(data_root, args.patch_size)

    ckpt_path = Path(args.checkpoint).expanduser()
    results_dir = ckpt_path.parent.parent / f'results-for-{ckpt_path.name}'

    if not args.skip_infer:
        if results_dir.exists():
            rmtree(str(results_dir))

        results_dir.mkdir(exist_ok=True)

        print('\nMaking inference ...')

        if args.model == 'wesup' and args.pixel:
            pixel_infer(patch_dir, checkpoint=args.checkpoint,
                        scales=(0.4,), output_dir=results_dir)
        else:
            infer(patch_dir, model_type=args.model, checkpoint=args.checkpoint,
                  input_size=(400, 400), output_dir=results_dir)

    output_name = f'combined-results-pixel-for-{ckpt_path.name}' if args.pixel else f'combined-results-for-{ckpt_path.name}'
    output_dir = results_dir.parent / output_name
    output_dir.mkdir(exist_ok=True)

    def combine_single(patches, original_size):
        height, width = original_size
        patch_size = cv2.imread(str(patches[0])).shape[0]

        ext_height = math.ceil(height / patch_size) * patch_size
        ext_width = math.ceil(width / patch_size) * patch_size
        final = np.zeros((ext_height, ext_width))

        for patch_path in patches:
            patch = cv2.imread(str(patch_path), cv2.IMREAD_GRAYSCALE)
            _, x, y = patch_path.name.replace(patch_path.suffix, '').split('-')
            x, y = int(x), int(y)
            final[y:y + patch_size, x:x + patch_size] = patch

        return final[:height, :width]

    with open(patch_dir / 'info.csv') as fp:
        reader = csv.reader(fp)
        info = [(stem, int(h), int(w)) for stem, h, w in reader]

    print('\nCombining predictions ...')
    pos_examples = []
    neg_examples = []
    for index, inf in tqdm(enumerate(info), total=len(info)):
        stem, height, width = inf

        patches = list(results_dir.glob(f'{index}-*'))
        combined = combine_single(patches, (height, width))
        dest = str(output_dir / f'{stem}{patches[0].suffix}')

        cv2.imwrite(dest, combined)

        if stem.startswith('positive-'):
            pos_examples.append(combined)
        else:
            neg_examples.append(combined)

    print(f'Combined results saved to {output_dir}.')

    executor = Parallel(2)
    gt_dir = data_root / 'masks'

    def read_mask(p):
        return cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)

    print('\nEvaluating positive OA and Dice ...')
    pos_paths = sorted(output_dir.glob('positive-*.png'))
    pos_examples = executor(delayed(read_mask)(pos_path)
                            for pos_path in pos_paths)
    pos_gts = executor(delayed(read_mask)(gt_path)
                       for gt_path in sorted(gt_dir.glob('positive-*.png')))
    compute_metrics(
        pos_examples, pos_gts, pos_paths, negative=False)

    print('\nEvaluating negative OA and Dice ...')
    neg_paths = sorted(output_dir.glob('negative-*.png'))
    neg_examples = executor(delayed(read_mask)(neg_path)
                            for neg_path in neg_paths)
    neg_gts = executor(delayed(read_mask)(gt_path)
                       for gt_path in sorted(gt_dir.glob('negative-*.png')))
    compute_metrics(
        neg_examples, neg_gts, neg_paths, negative=True)
