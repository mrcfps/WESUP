"""Test script to GlaS dataset."""

import argparse
from pathlib import Path
from shutil import rmtree

import torch

from infer import infer
from models import initialize_trainer


def test(ckpt_path, model_type='wesup', input_size=None, scales=(0.5,), device='cpu'):

    ckpt_path = Path(ckpt_path)
    trainer = initialize_trainer(model_type, device=device)
    trainer.load_checkpoint(ckpt_path)

    record_dir = ckpt_path.parent.parent

    if input_size is not None:
        results_dir = record_dir / 'results'
    else:
        results_dir = record_dir / f'results-{len(scales)}scale'

    if not results_dir.exists():
        results_dir.mkdir()

    try:
        print('\nTesting on test set A ...')
        data_dir = Path.home() / 'data' / 'GLAS_all' / 'testA'
        output_dir = results_dir / 'testA'
        infer(trainer, data_dir, output_dir, input_size, scales, device=device)

        print('\nTesting on test set B ...')
        data_dir = Path.home() / 'data' / 'GLAS_all' / 'testB'
        output_dir = results_dir / 'testB'
        infer(trainer, data_dir, output_dir, input_size, scales, device=device)
    finally:
        rmtree('models_ckpt', ignore_errors=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', default='wesup',
                        help='Which model to use')
    parser.add_argument('--input-size', help='Input size for model')
    parser.add_argument('--scales', default='0.6,0.55,0.5,0.45,0.4',
                        help='Optional multiscale inference')
    parser.add_argument('-c', '--checkpoint', help='Path to checkpoint')
    parser.add_argument('-d', '--device', default=('cuda' if torch.cuda.is_available() else 'cpu'),
                        help='Which device to use')
    args = parser.parse_args()

    input_size = None
    if args.input_size is not None:
        input_size = [int(s) for s in args.input_size.split(',')]
    scales = tuple(float(s) for s in args.scales.split(','))

    test(args.checkpoint, model_type=args.model, input_size=input_size,
         scales=scales, device=args.device)
