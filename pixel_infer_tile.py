import argparse
import os
from pathlib import Path

import torch
import torchvision.transforms.functional as TF

import numpy as np
from tqdm import tqdm
from PIL import Image
from skimage.io import imread, imsave

from models.wesup import WESUPPixelInference
from infer_tile import divide_image_to_patches
from infer_tile import combine_patches_to_image


def read_image_info(img_path):
    return img_path.name, imread(str(img_path)).shape[:2]


def fuse_prediction_and_save(multiple_preds, info):
    name, target_size = info
    fused_pred = sum(multiple_preds) / len(multiple_preds)
    imsave(output_dir / name, fused_pred.round().astype('uint8') * 255, check_contrast=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_root')
    parser.add_argument('-c', '--checkpoint', required=True)
    parser.add_argument('-p', '--patch-size', type=int, default=300)
    parser.add_argument('-o', '--output')
    args = parser.parse_args()

    data_root = Path(args.data_root).expanduser()
    ckpt_path = Path(args.checkpoint).expanduser()

    device = 'cuda'
    if args.output:
        output_dir = Path(args.output).expanduser()
    else:
        output_dir = ckpt_path.parent.parent / f'results-pixel-tile-{args.patch_size}' / data_root.name

    if not output_dir.exists():
        os.makedirs(output_dir)

    model = WESUPPixelInference().to(device)
    model.load_state_dict(torch.load(args.checkpoint)['model_state_dict'])

    with torch.no_grad():
        print(f'Making inference ...')
        img_paths = list((data_root / 'images').iterdir())

        for img_path in tqdm(img_paths):
            img = imread(str(img_path))
            patches = divide_image_to_patches(img, args.patch_size)
            predictions = []

            for patch in patches:
                input_ = TF.to_tensor(Image.fromarray(patch)).to(device)
                pred = model(input_.unsqueeze(0))
                pred = pred.detach().cpu().numpy()[..., 1]
                predictions.append(np.expand_dims(pred, 0))

            predictions = np.concatenate(predictions)
            final_pred = combine_patches_to_image(predictions, img.shape[0], img.shape[1])
            imsave(output_dir / img_path.name,
                   final_pred.round().astype('uint8') * 255,
                   check_contrast=False)
