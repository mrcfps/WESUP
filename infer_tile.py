"""
Inference module for window-based strategy.
"""

import argparse
import glob
import math
import os
import os.path as osp
from itertools import product

import numpy as np
import torch
import torchvision.transforms.functional as TF

from tqdm import tqdm
from PIL import Image
from skimage.io import imread

from models import initialize_model


def build_cli_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_path', help='Path to dataset')
    parser.add_argument('-m', '--model', default='mild',
                        help='Which model to use')
    parser.add_argument('-c', '--checkpoint', required=True,
                        help='Path to checkpoint')
    parser.add_argument('-p', '--patch-size', type=int, required=True,
                        help='Size of input patches')
    parser.add_argument('-d', '--device', default=('cuda' if torch.cuda.is_available() else 'cpu'),
                        help='Which device to use')
    parser.add_argument('-o', '--output',
                        help='Path to store inference results')

    return parser


def _list_images(path):
    """Glob all images within a directory."""

    images = []
    for ext in ("jpg", "jpeg", "png", "bmp"):
        images.extend(glob.glob(osp.join(path, f"*.{ext}")))
    return sorted(images)


def _get_top_left_coordinates(height, width, patch_size):
    """Calculate coordinates of top-left corners for patches."""

    n_h = math.ceil(height / patch_size)
    n_w = math.ceil(width / patch_size)
    tops = np.linspace(0, height - patch_size, n_h, dtype=int)
    lefts = np.linspace(0, width - patch_size, n_w, dtype=int)

    return product(tops, lefts)


def divide_image_to_patches(img, patch_size):
    """
    Divide a large image (mask) to patches with (possibly overlapping) tile strategy.

    Args:
        img: input image of shape (H, W, 3)
        patch_size: target size of patches

    Returns:
        patches: patches of shape (N, patch_size, patch_size, 3)
    """

    assert len(img.shape) == 3 and img.shape[-1] == 3

    height, width, n_channels = img.shape
    coordinates = _get_top_left_coordinates(height, width, patch_size)

    patches = []

    for top, left in coordinates:
        patches.append(img[top:top + patch_size, left:left + patch_size])

    return np.array(patches).astype('uint8')


def combine_patches_to_image(patches, target_height, target_width):
    """Combine patches back to a single image (mask).

    Args:
        patches: predicted patches of shape (N, H, W, C) or (N, H, W)
        target_height: target height of combined image
        target_width: target width of combined image
    
    Returns:
        combined: combined output of shape (H, W, C) or (H, W)
    """

    counter = 0
    patch_size = patches.shape[1]
    coordinates = _get_top_left_coordinates(target_height, target_width, patch_size)

    if len(patches.shape) == 3:  # channel dimension is missing
        patches = np.expand_dims(patches, -1)

    # The last channel is the number of overlapping patches for a given pixel,
    # used for averaging predictions from multiple windows.
    combined = np.zeros((target_height, target_width, patches.shape[-1] + 1))

    for top, left in coordinates:
        patch = combined[top:top + patch_size, left:left + patch_size, :-1]
        overlaps = combined[top:top + patch_size, left:left + patch_size, -1:]
        patch = (patch * overlaps + patches[counter]) / (overlaps + 1)
        combined[top:top + patch_size, left:left + patch_size, :-1] = patch
        overlaps += 1.
        counter += 1

    return np.squeeze(combined[..., :-1])


def predict(model, img_path, patch_size, device='cpu'):
    """Predict on a single input image.

    Arguments:
        model: inference model (should be a `torch.nn.Module`)
        img_path: instance of `torch.utils.data.Dataset`
        patch_size: patch size when feeding into network
        device: target device

    Returns:
        predictions: list of model predictions of size (H, W)
    """

    img = imread(img_path)
    patches = divide_image_to_patches(img, patch_size)
    predictions = []

    for patch in patches:
        input_ = TF.to_tensor(Image.fromarray(patch)).to(device).unsqueeze(0)
        prediction = model.postprocess(model(input_))
        prediction = prediction.detach().cpu().numpy()
        predictions.append(prediction[..., np.newaxis])

    predictions = np.concatenate(predictions)

    return combine_patches_to_image(predictions, img.shape[0], img.shape[1])


def save_predictions(predictions, img_paths, output_dir='predictions'):
    """Save predictions to disk.

    Args:
        predictions: model predictions of size (N, H, W)
        img_paths: list of paths to input images
        output_dir: path to output directory
    """

    print(f'\nSaving prediction to {output_dir} ...')

    if not osp.exists(output_dir):
        os.mkdir(output_dir)

    for pred, img_path in tqdm(zip(predictions, img_paths), total=len(predictions)):
        img_name = osp.basename(img_path)
        pred = pred.astype('uint8')
        Image.fromarray(pred * 255).save(osp.join(output_dir, img_name))


def infer(model, data_dir, patch_size, output_dir=None, device='cpu'):
    """Making inference on a directory of images with given model checkpoint."""

    if output_dir is not None and not osp.exists(output_dir):
        os.mkdir(output_dir)

    img_paths = _list_images(osp.join(data_dir, 'images'))

    print(f'Predicting {len(img_paths)} images from {data_dir} ...')
    predictions = [
        predict(model, img_path, patch_size, device=device)
        for img_path in tqdm(img_paths)
    ]

    if output_dir is not None:
        save_predictions(predictions, img_paths, output_dir)

    return predictions


if __name__ == '__main__':
    parser = build_cli_parser()
    args = parser.parse_args()

    device = args.device
    output_dir = args.output
    if output_dir is None:
        output_dir = osp.abspath(osp.join(osp.dirname(args.checkpoint), '..', 'results'))

    ckpt = torch.load(args.checkpoint)
    model = initialize_model(args.model, checkpoint=ckpt)
    model = model.to(device).eval()

    infer(model, args.dataset_path, args.patch_size,
          output_dir=output_dir, device=device)
