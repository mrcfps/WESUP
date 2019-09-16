"""Test script to GlaS dataset."""

import argparse
import os
import os.path as osp
from shutil import rmtree

import torch

from infer import prepare_model, infer


def test(ckpt_path, model_type='wessup', input_size=None,
         scales=(0.5,), num_workers=4, device='cpu'):
    record_dir = osp.abspath(osp.join(ckpt_path, '..', '..'))

    if input_size is not None:
        results_dir = osp.join(record_dir, 'results')
    else:
        results_dir = osp.join(record_dir, f'results-{len(scales)}scale')

    if not osp.exists(results_dir):
        os.mkdir(results_dir)

    model = prepare_model(model_type, ckpt_path, device=device)

    try:
        print('\nTesting on test set A ...')
        data_dir = osp.join('data/GLAS_all', 'testA')
        output_dir = osp.join(results_dir, 'testA')
        infer(model, data_dir, output_dir, input_size, scales, num_workers=num_workers, device=device)

        print('\nTesting on test set B ...')
        data_dir = osp.join('data/GLAS_all', 'testB')
        output_dir = osp.join(results_dir, 'testB')
        infer(model, data_dir, output_dir, input_size, scales, num_workers=num_workers, device=device)
    finally:
        rmtree('models_ckpt', ignore_errors=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', default='wessup',
                        help='Which model to use')
    parser.add_argument('--input-size', help='Input size for model')
    parser.add_argument('--scales', default='0.6,0.55,0.5,0.45,0.4',
                        help='Optional multiscale inference')
    parser.add_argument('-c', '--checkpoint', help='Path to checkpoint')
    parser.add_argument('-d', '--device', default=('cuda' if torch.cuda.is_available() else 'cpu'),
                        help='Which device to use')
    parser.add_argument('-j', '--jobs', type=int, default=os.cpu_count(),
                        help='Number of CPUs to use for preprocessing')
    args = parser.parse_args()

    input_size = None
    if args.input_size is not None:
        input_size = [int(s) for s in args.input_size.split(',')]
    scales = tuple(float(s) for s in args.scales.split(','))

    test(args.checkpoint, model_type=args.model, input_size=input_size,
         scales=scales, num_workers=args.jobs, device=args.device)
