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
import pydensecrf.densecrf as dcrf
from tqdm import tqdm
from PIL import Image
from skimage.transform import resize

import config
from utils.data import SegmentationDataset
from utils.metrics import accuracy
from utils.metrics import detection_f1
from utils.metrics import dice
from utils.metrics import object_dice
from utils.metrics import object_hausdorff
from utils.preprocessing import preprocess_superpixels

warnings.filterwarnings('ignore')


def compute_mask_with_superpixel_prediction(sp_pred, sp_maps):
    """
    Compute patch mask from superpixel predictions.

    Arguments:
        sp_pred: superpixel predictions with size (N, n_classes)
        sp_maps: superpixel maps with size (N, H, W)

    Returns:
        pred_mask: segmentation pprediction with size (H, W, n_classes)
    """

    # flatten sp_maps to one channel
    sp_maps = sp_maps.argmax(dim=0)

    # initialize prediction mask
    height, width = sp_maps.size()
    pred_mask = torch.zeros(height, width, sp_pred.size(1))
    pred_mask = pred_mask.to(sp_maps.device)

    for sp_idx in range(sp_maps.max().item() + 1):
        pred_mask[sp_maps == sp_idx] = sp_pred[sp_idx]

    return pred_mask


def predict(model, dataloader):
    """Predict on a directory of images.

    Arguments:
        model: inference model (should be a `torch.nn.Module`)
        dataloader: PyTorch DataLoader instance

    Returns:
        predictions: a list of predicted mask, each of which is a tensor of
            size (H, W, n_classes)
    """

    model.eval()
    device = next(model.parameters()).device

    predictions = []

    for data in tqdm(dataloader):
        img, segments = data[:2]

        img = img.to(device)
        segments = segments.to(device).squeeze()
        sp_maps = preprocess_superpixels(segments)

        with torch.no_grad():
            sp_pred = model(img, sp_maps)

        pred_mask = compute_mask_with_superpixel_prediction(sp_pred, sp_maps)
        predictions.append(pred_mask)

    return predictions


def crf_postprocess(img, pred_mask):
    """Post-processing using dense CRF.

    Arguments:
        img: original PIL image
        pred_mask: prediction mask array with shape (H, W, n_classes)

    Returns
    """

    h, w = pred_mask.shape[0], pred_mask.shape[1]
    img = np.array(img.resize((w, h), resample=Image.BILINEAR))
    d = dcrf.DenseCRF2D(w, h, 2)

    U = -np.log(pred_mask).transpose(2, 0, 1).reshape((2, -1))
    U = np.ascontiguousarray(U)
    img = np.ascontiguousarray(img)

    d.setUnaryEnergy(U)
    d.addPairwiseGaussian(sxy=1, compat=10)
    d.addPairwiseBilateral(sxy=10, srgb=10, rgbim=img, compat=10)

    Q = d.inference(5)
    Q = np.argmax(np.array(Q), axis=0).reshape((h, w))

    return Q


def infer(model, data_dir, viz_dir=None, use_crf=False, epoch=None, num_workers=4):
    """Making inference on a directory of images

    Arguments:
        model: inference model (should be a `torch.nn.Module`)
        data_dir: path to dataset, which should contains at least a subdirectory `images`
            with all images to be predicted
        viz_dir: path to store visualization and metrics results
        use_crf: whether to apply CRF post-processing
        epoch: current training epoch
        num_workers: number of workers to load data
    """

    dataset = SegmentationDataset(data_dir, rescale_factor=config.RESCALE_FACTOR, train=False)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=num_workers)

    print(f'Predicting {len(dataset)} images ...')
    predictions = predict(model, dataloader)

    if viz_dir is not None and not os.path.exists(viz_dir):
        os.mkdir(viz_dir)

    # whether to compute metrics
    evaluate = dataset.mask_paths is not None

    if evaluate:
        # record metrics of each image
        metrics = defaultdict(list)

    print('Computing metrics ...')
    for idx, pred_mask in tqdm(enumerate(predictions), total=len(predictions)):
        orig_img = Image.open(dataset.img_paths[idx])
        pred_mask = pred_mask.cpu().numpy()

        if use_crf:
            pred_mask = crf_postprocess(orig_img, pred_mask)
        else:
            pred_mask = pred_mask.argmax(axis=-1)

        # resize mask to match the size of original image
        pred_mask = resize(pred_mask, (orig_img.height, orig_img.width), order=0, preserve_range=True)
        pred_mask = pred_mask.astype('uint8')

        if evaluate:
            mask = np.array(Image.open(dataset.mask_paths[idx]))
            metrics['accuracy'].append(accuracy(pred_mask, mask))
            metrics['detection_f1'].append(detection_f1(pred_mask, mask))
            metrics['dice'].append(dice(pred_mask, mask))
            metrics['object_dice'].append(object_dice(pred_mask, mask))
            metrics['object_hausdorff'].append(object_hausdorff(pred_mask, mask))

        if viz_dir is not None:
            img_name = os.path.basename(dataset.img_paths[idx])
            extname = os.path.splitext(img_name)[-1]
            pred_name = img_name.replace(extname, f'.pred{"-" + str(epoch) if epoch else ""}{extname}')

            # save original image
            orig_img.save(os.path.join(viz_dir, img_name))

            # save prediction
            Image.fromarray(pred_mask * 255).save(os.path.join(viz_dir, pred_name))

            # save ground truth if any
            if evaluate:
                mask_name = img_name.replace(extname, f'.gt{extname}')
                Image.fromarray(mask * 255).save(os.path.join(viz_dir, mask_name))

    if viz_dir is not None:
        print(f'Prediction has been saved to {viz_dir}.')

    if evaluate:
        metrics = {
            k: np.mean(v)
            for k, v in metrics.items()
        }

        print('Mean Overall Accuracy:', metrics['accuracy'])
        print('Mean Dice:', metrics['dice'])
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

    try:
        wessup = import_module('wessup_ckpt')
        ckpt = torch.load(args.checkpoint, map_location=device)
        model = wessup.Wessup(ckpt['backbone'])
        model = model.to(device)
        model.load_state_dict(ckpt['model_state_dict'])
        print(f'Loaded checkpoint from {args.checkpoint}.')
        infer(model, args.dataset_path, args.output, epoch=ckpt['epoch'], num_workers=args.jobs)
    finally:
        os.remove('wessup_ckpt.py')
