"""
Utilities for recording multiple runs of experiments.
"""

import matplotlib
matplotlib.use('Agg')

import json
import os
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt

import config


def prepare_record_dir():
    """Create new record directory and return its path."""

    if not os.path.exists('records'):
        os.mkdir('records')

    record_dir = os.path.join(
        'records', datetime.now().strftime('%Y%m%d-%I%M-%p'))
    os.mkdir(record_dir)
    os.mkdir(os.path.join(record_dir, 'checkpoints'))

    return record_dir


def save_params(record_dir, args, fname='params'):
    """Save experiment parameters to record directory."""

    args = vars(args)

    # save all parameters in config.py
    for cfg_key in dir(config):
        if not cfg_key.startswith('__'):
            args[cfg_key.lower()] = getattr(config, cfg_key)

    num_of_runs = len([fn for fn in os.listdir(record_dir) if fn.startswith(fname)])
    fname = f'{fname}-{num_of_runs}.json'

    with open(os.path.join(record_dir, fname), 'w') as fp:
        json.dump(args, fp, indent=4)


def plot_learning_curves(history_path):
    """Read history csv file and plot learning curves."""

    history = pd.read_csv(history_path)

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
        plt.savefig(os.path.join(os.path.dirname(history_path), f'{key}.png'))
