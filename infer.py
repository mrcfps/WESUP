"""
Inference module.
"""

import warnings
from math import ceil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import fire
from tqdm import tqdm
from PIL import Image
from skimage.morphology import opening

from models import initialize_trainer
from utils.data import SegmentationDataset

warnings.filterwarnings('ignore')


def predict_single_image(trainer, img, mask, output_size):
    input_, target = trainer.preprocess(img, mask.long())

    with torch.no_grad():
        pred = trainer.model(input_)

    pred, _ = trainer.postprocess(pred, target)
    pred = pred.float().unsqueeze(0)
    pred = F.interpolate(pred, size=output_size, mode='nearest')

    return pred


def predict(trainer, dataset, input_size=None, scales=(0.5,),
            num_workers=4, device='cpu'):
    """Predict on a directory of images.

    Arguments:
        trainer: trainer instance (subclass of `models.base.BaseTrainer`)
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
    print(f'\nPredicting {len(dataset)} images with {size_info} ...')

    predictions = []
    for data in tqdm(dataloader, total=len(dataset)):
        img = data[0].to(device)
        mask = data[1].to(device).float()

        # original spatial size of input image (height, width)
        orig_size = (img.size(2), img.size(3))

        if input_size is not None:
            img = F.interpolate(img, size=input_size, mode='bilinear')
            mask = F.interpolate(mask, size=input_size, mode='nearest')
            prediction = predict_single_image(trainer, img, mask, orig_size)
        else:
            multiscale_preds = []
            for scale in scales:
                target_size = [ceil(size * scale) for size in orig_size]
                img = F.interpolate(img, size=target_size, mode='bilinear')
                mask = F.interpolate(mask, size=target_size, mode='nearest')
                multiscale_preds.append(
                    predict_single_image(trainer, img, mask, orig_size))

            prediction = torch.cat(multiscale_preds).mean(dim=0).round()

        prediction = prediction.squeeze().cpu().numpy()

        # apply morphology postprocessing (i.e. opening) for multi-scale inference
        if input_size is None and len(scales) > 1:
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

    output_dir = Path(output_dir)
    if not output_dir.exists():
        output_dir.mkdir()

    for pred, img_path in tqdm(zip(predictions, dataset.img_paths), total=len(predictions)):
        pred = pred.astype('uint8')
        Image.fromarray(pred * 255).save(output_dir / img_path.name)


def infer(trainer, data_dir, output_dir=None, input_size=None,
          scales=(0.5,), num_workers=4, device='cpu'):
    """Making inference on a directory of images with given model checkpoint."""

    trainer.model.eval()
    dataset = SegmentationDataset(data_dir, train=False)

    predictions = predict(trainer, dataset, input_size=input_size, scales=scales,
                          num_workers=num_workers, device=device)

    if output_dir is not None:
        save_predictions(predictions, dataset, output_dir)

    return predictions


def main(data_dir, model_type='wesup', checkpoint=None, output_dir=None,
         input_size=None, scales=(0.5,), num_workers=4, device=None):
    if output_dir is None and checkpoint is not None:
        checkpoint = Path(checkpoint)
        output_dir = checkpoint.parent.parent / 'results'
        if not output_dir.exists():
            output_dir.mkdir()

    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    trainer = initialize_trainer(model_type, device=device)
    if checkpoint is not None:
        trainer.load_checkpoint(checkpoint)

    infer(trainer, data_dir, output_dir, input_size=input_size,
          scales=scales, num_workers=num_workers, device=device)


if __name__ == '__main__':
    fire.Fire(main)
