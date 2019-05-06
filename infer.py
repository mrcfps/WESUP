"""
Inference module.
"""

import argparse
import csv
import os
import warnings
from collections import defaultdict
from importlib import import_module
from shutil import copyfile

import torch
from torch.utils.data import DataLoader

import numpy as np
from tqdm import tqdm
from PIL import Image

import config
from utils.data import WholeImageDataset
from utils.metrics import detection_f1
from utils.metrics import object_dice
from utils.metrics import object_hausdorff
from utils.tile import combine_patches_to_image
from utils.preprocessing import preprocess_superpixels

warnings.filterwarnings('ignore')


def compute_mask_with_superpixel_prediction(sp_pred, sp_maps):
    """
    Compute patch mask from superpixel predictions.

    Arguments:
        sp_pred: superpixel predictions with size (N, n_classes)
        sp_maps: superpixel maps with size (N, H, W)
    """

    sp_pred = sp_pred.argmax(dim=-1)

    # flatten sp_maps to one channel
    sp_maps = sp_maps.argmax(dim=0)

    # initialize prediction mask
    pred_mask = torch.zeros_like(sp_maps).to(sp_maps.device)

    for sp_idx in range(sp_maps.max().item() + 1):
        pred_mask[sp_maps == sp_idx] = sp_pred[sp_idx]

    return pred_mask


def test_whole_images(model, data_dir, viz_dir=None, epoch=None,
                      evaluate=True, num_workers=4):
    """Making inference on a directory of images.

    Arguments:
        model: inference model (should be a `torch.nn.Module`)
        data_dir: path to dataset, which should contains at least a subdirectory `images`
            with all images to be predicted
        viz_dir: path to store visualization and metrics results
        epoch: current training epoch
        evaluate: whether to compute metrics
        num_workers: number of workers to load data
    """

    model.eval()
    device = next(model.parameters()).device
    dataset = WholeImageDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=num_workers)

    if viz_dir is not None and not os.path.exists(viz_dir):
        os.mkdir(viz_dir)

    if evaluate:
        # record metrics of each image
        metrics = defaultdict(list)

    print('\nTesting whole images ...')
    pred_masks = []
    for patch_idx, data in tqdm(enumerate(dataloader), total=len(dataset)):
        # identify which image this patch belongs to
        img_idx = dataset.patch2img(patch_idx)

        patch, segments = data
        patch = patch.to(device)
        segments = segments.to(device).squeeze()
        sp_maps = preprocess_superpixels(segments)

        sp_pred = model(patch, sp_maps)
        pred_masks.append(
            compute_mask_with_superpixel_prediction(sp_pred, sp_maps).cpu().numpy()
        )

        if len(pred_masks) == dataset.patches_nums[img_idx]:
            img = Image.open(dataset.img_paths[img_idx])

            # all patches of an image have been predicted, so combine the predictions
            pred_masks = np.expand_dims(np.array(pred_masks), -1)
            whole_pred = combine_patches_to_image(pred_masks,
                                                  dataset.patches_grids[img_idx],
                                                  (img.height, img.width),
                                                  config.INFER_STRIDE)
            whole_pred = whole_pred.squeeze().round().astype('uint8')

            if evaluate:
                ground_truth = dataset.masks[img_idx]
                metrics['detection_f1'].append(detection_f1(whole_pred, ground_truth))
                metrics['object_dice'].append(object_dice(whole_pred, ground_truth))
                metrics['object_hausdorff'].append(object_hausdorff(whole_pred, ground_truth))

            if viz_dir is not None:
                img_name = os.path.basename(dataset.img_paths[img_idx])
                extname = os.path.splitext(img_name)[-1]
                pred_name = img_name.replace(extname, f'.pred{"-" + str(epoch) if epoch else ""}{extname}')

                img.save(os.path.join(viz_dir, img_name))
                Image.fromarray(whole_pred * 255).save(os.path.join(viz_dir, pred_name))

                if dataset.masks is not None:
                    mask_name = img_name.replace(extname, f'.gt{extname}')
                    Image.fromarray(dataset.masks[img_idx] * 255).save(os.path.join(viz_dir, mask_name))

            pred_masks = []

    if evaluate:
        metrics = {
            k: np.mean(v)
            for k, v in metrics.items()
        }

        print('Mean Detection F1:', metrics['detection_f1'])
        print('Mean Object Dice:', metrics['object_dice'])
        print('Mean Object Hausdorff:', metrics['object_hausdorff'])

        if viz_dir is not None:
            metrics_path = os.path.join(viz_dir, 'metrics.csv')
            if not os.path.exists(metrics_path):
                with open(metrics_path, 'w') as fp:
                    writer = csv.writer(fp)
                    writer.writerow(['epoch'] + list(metrics.keys()))
                    writer.writerow([epoch] + list(metrics.values()))
            else:
                with open(metrics_path, 'a') as fp:
                    writer = csv.writer(fp)
                    writer.writerow([epoch] + list(metrics.values()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_path', help='Path to dataset')
    parser.add_argument('-c', '--checkpoint', required=True, help='Path to checkpoint')
    parser.add_argument('--wessup-module', help='Path to wessup module (.py file)')
    parser.add_argument('--no-gpu', action='store_true', default=False,
                        help='Whether to avoid using gpu')
    parser.add_argument('-o', '--output',
                        help='Path to store visualization and metrics result')
    parser.add_argument('-j', '--jobs', type=int, default=int(os.cpu_count() / 2),
                        help='Number of CPUs to use for preprocessing')

    args = parser.parse_args()

    device = 'cpu' if args.no_gpu or not torch.cuda.is_available() else 'cuda'

    wessup_module = args.wessup_module
    if wessup_module is None:
        wessup_module = os.path.join(args.checkpoint, '..', '..', 'source', 'wessup.py')
        wessup_module = os.path.abspath(wessup_module)
    copyfile(wessup_module, 'wessup_ckpt.py')
    wessup = import_module('wessup_ckpt')

    ckpt = torch.load(args.checkpoint, map_location=device)
    model = wessup.Wessup(ckpt['backbone'])
    model.to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f'Loaded checkpoint from {args.checkpoint}.')

    test_whole_images(model, args.dataset_path, args.output,
                      epoch=ckpt['epoch'], evaluate=True, num_workers=args.jobs)

    os.remove('wessup_ckpt.py')
