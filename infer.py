"""
Inference module.
"""

import argparse
import csv
import os
import warnings
from importlib import import_module
from shutil import copytree, rmtree

import torch
import torch.nn.functional as F

import numpy as np
from tqdm import tqdm
from PIL import Image
from skimage.io import imread

from utils.metrics import accuracy
from utils.metrics import detection_f1
from utils.metrics import dice
from utils.metrics import object_dice
from utils.metrics import object_hausdorff

warnings.filterwarnings('ignore')


def build_cli_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_path', help='Path to dataset')
    parser.add_argument('-m', '--model', default='wessup', choices=['wessup', 'cdws', 'wtp'],
                        help='Which model to use')
    parser.add_argument('-b', '--batch-size', type=int, default=1,
                        help='Batch size for inference')
    parser.add_argument('-c', '--checkpoint',
                        help='Path to checkpoint')
    parser.add_argument('--no-gpu', action='store_true', default=False,
                        help='Whether to avoid using gpu')
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
    models_dir = os.path.join(ckpt_path, '..', '..', 'source', 'models')
    models_path = 'models_ckpt'
    if os.path.exists(models_dir):
        copytree(models_dir, models_path)
    else:
        # fall back to current models module
        models_path = 'models'

    models = import_module(models_path)

    if model_type == 'wessup':
        model = models.Wessup(checkpoint=checkpoint)
    elif model_type == 'cdws':
        model = models.CDWS(checkpoint=checkpoint)
    elif model_type == 'wtp':
        model = models.WhatsThePoint(checkpoint=checkpoint)
    else:
        raise ValueError(f'Unsupported model: {model_type}')

    return model.to(device).eval()


def predict(model, dataset, batch_size=1, num_workers=4, device='cpu'):
    """Predict on a directory of images.

    Arguments:
        model: inference model (should be a `torch.nn.Module`)
        dataset: instance of `torch.utils.data.Dataset`
        batch_size: mini-batch size for inference
        num_workers: number of workers to load data
        device: target device

    Returns:
        predictions: list of model predictions of size (H, W)
    """

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers)

    print(f'\nPredicting {len(dataset)} images ...')
    pbar = tqdm(total=len(dataset))
    predictions = []
    for data in dataloader:
        data = [datum.to(device) for datum in data]
        input_, target = model.preprocess(*data)

        with torch.no_grad():
            pred = model(input_)

        pred, _ = model.postprocess(pred, target)
        predictions.append(pred)

        pbar.update(data[0].size(0))

    return predictions


def resize_pred_to_original_size(predictions, dataset):
    """Resize each prediction back to its original size."""

    print('\nResizing predictions back to original sizes ...')

    results = []
    for pred, img_path in tqdm(zip(predictions, dataset.img_paths),
                               total=len(predictions)):
        img = Image.open(img_path)
        pred = pred.unsqueeze(0).float()
        pred = F.interpolate(pred, size=(img.height, img.width))
        results.append(pred.squeeze().cpu().numpy())

    return results


def save_predictions(predictions, dataset, output_dir='predictions'):
    """Save predictions to disk.

    Args:
        predictions: model predictions of size (N, H, W)
        dataset: dataset for prediction, used for naming the prediction output
        output_dir: path to output directory
    """

    print(f'\nSaving prediction to {output_dir} ...')

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    for pred, img_path in tqdm(zip(predictions, dataset.img_paths), total=len(predictions)):
        img_name = os.path.basename(img_path)
        pred = pred.astype('uint8')
        Image.fromarray(pred * 255).save(os.path.join(output_dir, img_name))


def report_metrics(metrics):
    print('Mean Overall Accuracy:', metrics['accuracy'])
    print('Mean Dice:', metrics['dice'])
    print('Mean Detection F1:', metrics['detection_f1'])
    print('Mean Object Dice:', metrics['object_dice'])
    print('Mean Object Hausdorff:', metrics['object_hausdorff'])


def infer(model, data_dir, output_dir=None, batch_size=1,
          num_workers=4, device='cpu'):
    """Making inference on a directory of images with given model checkpoint."""

    if output_dir is not None and not os.path.exists(output_dir):
        os.mkdir(output_dir)

    dataset = model.get_default_dataset(data_dir, train=False)

    predictions = predict(model, dataset, batch_size=batch_size,
                          num_workers=num_workers, device=device)
    predictions = resize_pred_to_original_size(predictions, dataset)

    if output_dir is not None:
        save_predictions(predictions, dataset, output_dir)

    if dataset.mask_paths is not None:
        targets = [imread(mask_path) for mask_path in dataset.mask_paths]
        metric_funcs = [accuracy, dice, detection_f1, object_dice, object_hausdorff]

        print('\nComputing metrics ...')
        metrics = model.evaluate(predictions, targets, metric_funcs, verbose=True)
        report_metrics(metrics)

        if output_dir is not None:
            metrics_path = os.path.join(output_dir, 'metrics.csv')
            with open(metrics_path, 'w') as fp:
                writer = csv.writer(fp)
                writer.writerow(metrics.keys())
                writer.writerow(metrics.values())


if __name__ == '__main__':
    parser = build_cli_parser()
    args = parser.parse_args()

    device = 'cpu' if args.no_gpu or not torch.cuda.is_available() else 'cuda'
    model = prepare_model(args.model, args.checkpoint, device=device)

    try:
        infer(model, args.dataset_path, output_dir=args.output,
              batch_size=args.batch_size, num_workers=args.jobs, device=device)
    finally:
        rmtree('models_ckpt', ignore_errors=True)
