"""
Inference module.
"""

import argparse
import os
import os.path as osp
import warnings
from math import ceil
from importlib import import_module
from shutil import copytree, rmtree

import numpy as np
import torch
import torch.nn.functional as F

from tqdm import tqdm
from PIL import Image
from skimage.morphology import opening

from utils.data import SegmentationDataset

warnings.filterwarnings('ignore')


def build_cli_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_path', help='Path to dataset')
    parser.add_argument('-m', '--model', default='wessup',
                        help='Which model to use')
    parser.add_argument('-c', '--checkpoint', required=True,
                        help='Path to checkpoint')
    parser.add_argument('--input-size', help='Input size for model')
    parser.add_argument('--scales', default='0.5', help='Optional multiscale inference')
    parser.add_argument('-d', '--device', default=('cuda' if torch.cuda.is_available() else 'cpu'),
                        help='Which device to use')
    parser.add_argument('-o', '--output', default='predictions',
                        help='Path to store visualization and metrics result')
    parser.add_argument('-j', '--jobs', type=int, default=os.cpu_count(),
                        help='Number of CPUs to use for preprocessing')

    return parser


def prepare_model(model_type, ckpt_path=None, device='cpu'):
    """Prepare model for inference."""

    checkpoint = None
    if ckpt_path is not None:
        checkpoint = torch.load(ckpt_path, map_location=device)
        print(f'Loaded checkpoint from {ckpt_path}.')

    # copy models module next to checkpoint directory (if present)
    models_dir = osp.abspath(osp.join(ckpt_path, '..', '..', 'source', 'models'))
    models_path = 'models_ckpt'
    if osp.exists(models_dir):
        if osp.exists(models_path):
            rmtree(models_path)
        copytree(models_dir, models_path)
    else:
        # fall back to current models module
        models_path = 'models'

    models = import_module(models_path)
    if hasattr(models, 'initialize_model'):
        model = models.initialize_model(model_type, checkpoint=checkpoint)
    elif model_type == 'wessup':
        model = models.Wessup(checkpoint=checkpoint)
    elif model_type == 'cdws':
        model = models.CDWS(checkpoint=checkpoint)
    elif model_type == 'sizeloss':
        model = models.SizeLoss(checkpoint=checkpoint)
    else:
        raise ValueError(f'Unsupported model: {model_type}')

    return model.to(device).eval()


def predict_single_image(model, img, mask, input_size, output_size, device='cpu'):
    input_, target = model.preprocess(img, mask.long(), device=device)

    with torch.no_grad():
        pred = model(input_)

    pred, _ = model.postprocess(pred, target)
    pred = pred.float().unsqueeze(0)
    pred = F.interpolate(pred, size=output_size, mode='nearest')

    return pred


def predict(model, dataset, input_size=None, scales=(0.5,), num_workers=4, device='cpu'):
    """Predict on a directory of images.

    Arguments:
        model: inference model (should be a `torch.nn.Module`)
        dataset: instance of `torch.utils.data.Dataset`
        input_size: spatial size of input image
        scales: rescale factors for multi-scale inference
        num_workers: number of workers to load data
        device: target device

    Returns:
        predictions: list of model predictions of size (H, W)
    """

    dataloader = torch.utils.data.DataLoader(dataset, num_workers=num_workers)

    size_info = f'input size {input_size}' if input_size else f'scales {scales}'
    print(f'\nPredicting {len(dataset)} images with scales {size_info} ...')

    predictions = []
    for data in tqdm(dataloader, total=len(dataset)):
        img = data[0].to(device)
        mask = data[1].to(device).float()

        # original spatial size of input image (height, width)
        orig_size = (img.size(2), img.size(3))

        if input_size is not None:
            img = F.interpolate(img, size=input_size, mode='bilinear')
            mask = F.interpolate(mask, size=input_size, mode='nearest')
            prediction = predict_single_image(model, img, mask, input_size, orig_size, device=device)
        else:
            multiscale_preds = []
            for scale in scales:
                target_size = [ceil(size * scale) for size in orig_size]
                img = F.interpolate(img, size=target_size, mode='bilinear')
                mask = F.interpolate(mask, size=target_size, mode='nearest')
                multiscale_preds.append(
                    predict_single_image(model, img, mask, target_size, orig_size, device=device))

            prediction = torch.cat(multiscale_preds).mean(dim=0).round()

        prediction = prediction.squeeze().cpu().numpy()

        # apply morphology postprocessing (i.e. opening)
        # when performing multiscale inference
        if len(scales) > 1:
            def get_selem(size):
                assert size % 2 == 1
                selem = np.zeros((size, size))
                center = int((size + 1) / 2)
                selem[center, :] = 1
                selem[:, center] = 1
                return selem
            prediction = opening(prediction, selem=get_selem(9))

        predictions.append(prediction)

    return predictions


def save_predictions(predictions, dataset, output_dir='predictions'):
    """Save predictions to disk.

    Args:
        predictions: model predictions of size (N, H, W)
        dataset: dataset for prediction, used for naming the prediction output
        output_dir: path to output directory
    """

    print(f'\nSaving prediction to {output_dir} ...')

    if not osp.exists(output_dir):
        os.mkdir(output_dir)

    for pred, img_path in tqdm(zip(predictions, dataset.img_paths), total=len(predictions)):
        img_name = osp.basename(img_path)
        pred = pred.astype('uint8')
        Image.fromarray(pred * 255).save(osp.join(output_dir, img_name))


def infer(model, data_dir, output_dir=None, input_size=None,
          scales=(0.5,), num_workers=4, device='cpu'):
    """Making inference on a directory of images with given model checkpoint."""

    if output_dir is not None and not osp.exists(output_dir):
        os.mkdir(output_dir)

    dataset = SegmentationDataset(data_dir, train=False)

    predictions = predict(model, dataset, input_size=input_size, scales=scales,
                          num_workers=num_workers, device=device)

    if output_dir is not None:
        save_predictions(predictions, dataset, output_dir)


if __name__ == '__main__':
    parser = build_cli_parser()

    try:
        args = parser.parse_args()

        device = args.device
        input_size = None
        if args.input_size is not None:
            input_size = [int(s) for s in args.input_size.split(',')]
        scales = [float(s) for s in args.scales.split(',')]
        model = prepare_model(args.model, args.checkpoint, device=device)

        infer(model, args.dataset_path, output_dir=args.output, input_size=input_size,
              scales=scales, num_workers=args.jobs, device=device)
    finally:
        rmtree('models_ckpt', ignore_errors=True)
