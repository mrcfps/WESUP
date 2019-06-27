"""Test script to GlaS dataset."""

import argparse
import os
from shutil import rmtree

import torch

from infer import prepare_model, infer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_path', help='Path to dataset')
    parser.add_argument('-m', '--model', default='wessup', choices=['wessup', 'cdws'],
                        help='Which model to use')
    parser.add_argument('-c', '--checkpoint', help='Path to checkpoint')
    parser.add_argument('-d', '--device', default=('cuda' if torch.cuda.is_available() else 'cpu'),
                        help='Which device to use')
    parser.add_argument('-j', '--jobs', type=int, default=os.cpu_count(),
                        help='Number of CPUs to use for preprocessing')
    args = parser.parse_args()

    device = args.device
    record_dir = os.path.abspath(os.path.join(args.checkpoint, '..', '..'))
    results_dir = os.path.join(record_dir, 'results')
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    model = prepare_model(args.model, args.checkpoint, device=device)

    try:
        print('\nTesting on test set A ...')
        data_dir = os.path.join(args.dataset_path, 'testA')
        output_dir = os.path.join(results_dir, 'testA')
        infer(model, data_dir, output_dir, num_workers=args.jobs, device=device)

        print('\nTesting on test set B ...')
        data_dir = os.path.join(args.dataset_path, 'testB')
        output_dir = os.path.join(results_dir, 'testB')
        infer(model, data_dir, output_dir, num_workers=args.jobs, device=device)
    finally:
        rmtree('models_ckpt', ignore_errors=True)
