"""
Utilities for recording multiple runs of experiments.
"""

import matplotlib
matplotlib.use('Agg')

import json
import glob
import os
from datetime import datetime
from shutil import copyfile, copytree

import pandas as pd
import matplotlib.pyplot as plt

import config


def prepare_record_dir():
    """Create new record directory and return its path."""

    if not os.path.exists('records'):
        os.mkdir('records')

    record_dir = os.path.join(
        'records', datetime.now().strftime('%Y%m%d-%I%M-%p'))

    if not os.path.exists(record_dir):
        os.mkdir(record_dir)

    checkpoint_dir = os.path.join(record_dir, 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    return record_dir


def save_params(record_dir, args):
    """Save experiment parameters to record directory."""

    args = vars(args)
    params_dir = os.path.join(record_dir, 'params')

    if not os.path.exists(params_dir):
        os.mkdir(params_dir)

    # save all parameters in config.py
    for cfg_key in dir(config):
        if not cfg_key.startswith('__'):
            args[cfg_key.lower()] = getattr(config, cfg_key)

    num_of_runs = len([fn for fn in os.listdir(params_dir)])

    with open(os.path.join(params_dir, f'{num_of_runs}.json'), 'w') as fp:
        json.dump(args, fp, indent=4)


def copy_source_files(record_dir):
    """Copy all source scripts to record directory for reproduction."""

    source_dir = os.path.join(record_dir, 'source')
    if not os.path.exists(source_dir):
        os.mkdir(source_dir)

    for source_file in glob.glob('*.py'):
        copyfile(source_file, os.path.join(source_dir, source_file))

    copytree('utils', os.path.join(source_dir, 'utils'))


def plot_learning_curves(history_path):
    """Read history csv file and plot learning curves."""

    history = pd.read_csv(history_path)
    record_dir = os.path.dirname(history_path)
    curves_dir = os.path.join(record_dir, 'curves')

    if not os.path.exists(curves_dir):
        os.mkdir(curves_dir)

    for key in history.columns:
        if key.startswith('val_'):
            continue

        plt.figure(dpi=200)
        try:
            plt.plot(history[key])
            plt.plot(history['val_' + key])
        except KeyError:
            pass

        plt.title('Model ' + key)
        plt.ylabel(key.capitalize())
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Val'])
        plt.grid(True)
        plt.savefig(os.path.join(curves_dir, f'{key}.png'))
