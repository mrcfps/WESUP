"""
Training module.
"""

import logging
from shutil import rmtree

import fire

from models import initialize_trainer
from utils.metrics import accuracy
from utils.metrics import dice


def fit(dataset_path, model='wesup', **kwargs):
    # Initialize logger.
    logger = logging.getLogger('Train')
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler())

    trainer = initialize_trainer(model, logger=logger, **kwargs)

    try:
        trainer.train(dataset_path,
                      metrics=[accuracy, dice], **kwargs)
    finally:
        if kwargs.get('smoke'):
            rmtree(trainer.record_dir, ignore_errors=True)


if __name__ == '__main__':
    fire.Fire(fit)
