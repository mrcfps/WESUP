"""
Utilities for recording multiple runs of experiments.
"""

import os
import glob
import json
from shutil import copyfile, copytree, rmtree
from datetime import datetime
from pathlib import Path

import matplotlib
import pandas as pd
import matplotlib.pyplot as plt

matplotlib.use('Agg')


def prepare_record_dir():
    """Create new record directory and return its path."""

    record_root = Path.home() / 'records'
    if os.environ.get('RECORD_ROOT'):
        record_root = Path(os.environ.get('RECORD_ROOT')).expanduser()

    if not record_root.exists():
        record_root.mkdir()

    record_dir = record_root / datetime.now().strftime('%Y%m%d-%I%M-%p')

    if not record_dir.exists():
        record_dir.mkdir()

    checkpoint_dir = record_dir / 'checkpoints'
    if not checkpoint_dir.exists():
        checkpoint_dir.mkdir()

    return record_dir


def save_params(record_dir, params):
    """Save experiment parameters to record directory."""

    params_dir = record_dir / 'params'

    if not params_dir.exists():
        params_dir.mkdir()

    num_of_runs = len(list(params_dir.iterdir()))

    with open(params_dir / f'{num_of_runs}.json', 'w') as fp:
        json.dump(params, fp, indent=4)


def copy_source_files(record_dir):
    """Copy all source scripts to record directory for reproduction."""

    source_dir = record_dir / 'source'
    if source_dir.exists():
        rmtree(source_dir)
    source_dir.mkdir()

    for source_file in glob.glob('*.py'):
        copyfile(source_file, source_dir / source_file)

    copytree('utils', source_dir / 'utils')
    copytree('models', source_dir / 'models')
    copytree('scripts', source_dir / 'scripts')


def plot_learning_curves(history_path):
    """Read history csv file and plot learning curves."""

    history = pd.read_csv(history_path)
    record_dir = history_path.parent
    curves_dir = record_dir / 'curves'

    if not curves_dir.exists():
        curves_dir.mkdir()

    for key in history.columns:
        if key.startswith('val_'):
            if key.replace('val_', '') not in history.columns:
                # plot metrics computed only on validation phase
                plt.figure(dpi=200)
                plt.title('Model ' + key.replace('val_', ''))
                plt.plot(history[key])
                plt.ylabel(key.replace('val_', '').capitalize())
                plt.xlabel('Epoch')
                plt.grid(True)
                plt.savefig(curves_dir / f'{key}.png')
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
        plt.savefig(curves_dir / f'{key}.png')
        plt.close()
