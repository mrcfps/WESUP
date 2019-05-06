import argparse
import os
from importlib import import_module
from shutil import copyfile

import torch
from torchvision.models import vgg13

from infer import test_whole_images


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_path', help='Path to dataset')
    parser.add_argument('-c', '--checkpoint', required=True, help='Path to checkpoint')
    parser.add_argument('--no-gpu', action='store_true', default=False,
                        help='Whether to avoid using gpu')
    parser.add_argument('-j', '--jobs', type=int, default=int(os.cpu_count() / 2),
                        help='Number of CPUs to use for preprocessing')
    args = parser.parse_args()

    device = 'cpu' if args.no_gpu or not torch.cuda.is_available() else 'cuda'
    record_dir = os.path.abspath(os.path.join(args.checkpoint, '..', '..'))

    wessup_module = os.path.join(record_dir, 'source', 'wessup.py')
    copyfile(wessup_module, 'wessup_ckpt.py')
    wessup = import_module('wessup_ckpt')

    ckpt = torch.load(args.checkpoint, map_location=device)
    model = wessup.Wessup(ckpt['backbone'])
    model.to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f'Loaded checkpoint from {args.checkpoint}.')

    results_dir = os.path.join(record_dir, 'results')
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    print('\nTesting on test set A ...')
    data_dir = os.path.join(args.dataset_path, 'testA')
    output_dir = os.path.join(results_dir, 'testA')
    test_whole_images(model, data_dir, output_dir,
                      epoch=ckpt['epoch'], evaluate=True, num_workers=args.jobs)
    print(f'Results on test set A have been saved to {output_dir}.')

    print('\nTesting on test set B ...')
    data_dir = os.path.join(args.dataset_path, 'testB')
    output_dir = os.path.join(results_dir, 'testB')
    test_whole_images(model, data_dir, output_dir,
                      epoch=ckpt['epoch'], evaluate=True, num_workers=args.jobs)
    print(f'Results on test set B have been saved to {output_dir}.')

    os.remove('wessup_ckpt.py')
