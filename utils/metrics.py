import csv
import os
from collections import defaultdict

import torch


def superpixel_accuracy(sp_pred, sp_labels):
    """Superpixel classification accuracy."""

    return (sp_pred.argmax(dim=-1) == sp_labels).float().mean()


def superpixel_f1(sp_pred, sp_labels):
    """Superpixel classification F1 score."""

    pass


def pixel_accuracy(pred_mask, true_mask):
    """Overall pixel classification accuracy."""

    return (pred_mask == true_mask).float().mean()


def pixel_f1(input, target):
    """Overall pixel classification accuracy."""

    pass


class MetricsTracker:
    def __init__(self, save_path=None):
        self.history = defaultdict(list)
        self.save_path = save_path

    def train(self):
        self.is_train = True
        self.clear()

    def eval(self):
        self.is_train = False
        self.clear()

    def step(self, metrics):
        reports = list()
        for k, v in metrics.items():
            self.history[k if self.is_train else f'val_{k}'].append(v)
            reports.append('{} = {:.4f}'.format(k, v))

        return ', '.join(reports)

    def log(self):
        metrics = [sum(r) / len(r) for r in self.history.values()]
        print(', '.join(
            'Running {} = {:.4f}'.format(name, value)
            for name, value in zip(self.history.keys(), metrics)
        ))

        # no need to save history to csv file
        if self.save_path is None:
            return

        if not os.path.exists(self.save_path):
            # create a new csv file
            with open(self.save_path, 'w') as fp:
                writer = csv.writer(fp)
                writer.writerow(self.history.keys())
                writer.writerow(metrics)
        else:
            with open(self.save_path, 'a') as fp:
                writer = csv.writer(fp)
                writer.writerow(metrics)

    def clear(self):
        self.history.clear()
