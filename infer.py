"""
Inference module.
"""

import argparse
import os
import warnings
from collections import defaultdict

import torch
from torch.utils.data import DataLoader
from torchvision.models import vgg13

import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image

import config
from wessup import Wessup
from utils.data import WholeImageDataset
from utils.metrics import detection_f1
from utils.metrics import object_dice
from utils.metrics import object_hausdorff
from utils.tile import combine_patches_to_image

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
                      device='cpu', num_workers=4):
    """Making inference on a directory of images.

    Arguments:
        model: inference model (should be a `torch.nn.Module`)
        data_dir: path to dataset, which should contains at least a subdirectory `images`
            with all images to be predicted
        viz_dir: path to store visualization and metrics results
        epoch: current training epoch
        device: which device to run
        num_workers: number of workers to load data
    """

    model.eval()
    dataset = WholeImageDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=num_workers)

    if viz_dir is not None and not os.path.exists(viz_dir):
        os.mkdir(viz_dir)

    # whether to evaluate whole image predictions
    evaluate = dataset.masks is not None

    if evaluate:
        # Load previous metrics. If not present, create one.
        if os.path.exists(os.path.join(viz_dir, 'metrics.csv')):
            metrics = pd.read_csv(os.path.join(viz_dir, 'metrics.csv'))
        else:
            metrics = pd.DataFrame(
                columns=['epoch', 'detection_f1', 'object_dice', 'object_hausdorff'])

        # record metrics of each image
        running_metrics = defaultdict(list)
        running_metrics['epoch'] = [epoch] if epoch else None

    print('\nTesting whole images ...')
    pred_masks = []
    for patch_idx, data in tqdm(enumerate(dataloader), total=len(dataset)):
        # identify which image this patch belongs to
        img_idx = dataset.patch2img(patch_idx)

        patch, sp_maps = data
        patch = patch.to(device)
        sp_maps = sp_maps.to(device).squeeze()

        sp_pred = model(patch, sp_maps)
        pred_masks.append(
            compute_mask_with_superpixel_prediction(sp_pred, sp_maps).cpu().numpy()
        )

        if len(pred_masks) == dataset.patches_nums[img_idx]:
            img = Image.open(dataset.img_paths[img_idx])
            # all patches of an image have been predicted, so combine the predictions
            pred_masks = np.expand_dims(np.array(pred_masks), -1)
            whole_pred = combine_patches_to_image(pred_masks, (img.height, img.width),
                                                  config.INFER_STRIDE)
            whole_pred = whole_pred.squeeze().round().astype('uint8')

            if evaluate:
                ground_truth = dataset.masks[img_idx]
                running_metrics['detection_f1'].append(detection_f1(whole_pred, ground_truth))
                running_metrics['object_dice'].append(object_dice(whole_pred, ground_truth))
                running_metrics['object_hausdorff'].append(object_hausdorff(whole_pred, ground_truth))

            if viz_dir is not None:
                img_name = os.path.basename(dataset.img_paths[img_idx])
                extname = os.path.splitext(img_name)[-1]
                pred_name = img_name.replace(extname, f'.pred{"-" + str(epoch) if epoch else ""}{extname}')

                img.save(os.path.join(viz_dir, img_name))
                Image.fromarray(whole_pred * 255).save(os.path.join(viz_dir, pred_name))

                if evaluate:
                    mask_name = img_name.replace(extname, f'.gt{extname}')
                    Image.fromarray(dataset.masks[img_idx] * 255).save(os.path.join(viz_dir, mask_name))

            pred_masks = []

    if evaluate:
        metrics = metrics.append({
            k: np.mean(v)
            for k, v in running_metrics.items()
            if v is not None
        }, ignore_index=True)

        print('Mean Detection F1:', metrics.iloc[-1]['detection_f1'])
        print('Mean Object Dice:', metrics.iloc[-1]['object_dice'])
        print('Mean Object Hausdorff:', metrics.iloc[-1]['object_hausdorff'])

        if viz_dir is not None:
            metrics.to_csv(os.path.join(viz_dir, 'metrics.csv'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_path', help='Path to dataset')
    parser.add_argument('--no-gpu', action='store_true', default=False,
                        help='Whether to avoid using gpu')
    parser.add_argument('-o', '--output', default='viz',
                        help='Path to store visualization and metrics result')
    parser.add_argument('-j', '--jobs', type=int, default=int(os.cpu_count() / 2),
                        help='Number of CPUs to use for preprocessing ')
    parser.add_argument('-m', '--model', help='Path to saved model or checkpoint')

    args = parser.parse_args()

    device = 'cpu' if args.no_gpu and not torch.cuda.is_available() else 'cuda'
    wessup = Wessup(vgg13().features, device)

    if args.model:
        ckpt = torch.load(args.model)
        if 'model_state_dict' in ckpt:
            ckpt = ckpt['model_state_dict']
        wessup.load_state_dict(ckpt)
        print(f'Loaded model from {args.model}.')

    test_whole_images(wessup, args.dataset_path, args.output, device=device, num_workers=args.jobs)
