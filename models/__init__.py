import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.absolute()))

from .wesup import WESUP, WESUPConfig, WESUPTrainer


def initialize_trainer(model_type, **kwargs):
    """Initialize a trainer for model."""

    if model_type == 'wesup':
        kwargs = {**WESUPConfig().to_dict(), **kwargs}
        model = WESUP(**kwargs)
        trainer = WESUPTrainer(model, **kwargs)
    else:
        raise ValueError(f'Unsupported model: {model_type}')

    return trainer
