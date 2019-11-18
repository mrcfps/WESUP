import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.absolute()))

from .cdws_mil import CDWS, CDWSConfig, CDWSTrainer
from .mild_net import MILDNet, MILDNetConfig, MILDNetTrainer
from .wesup import WESUP, WESUPConfig, WESUPTrainer
from .wesupv2 import WESUPV2, WESUPV2Config, WESUPV2Trainer
from .sizeloss import SizeLoss, SizeLossConfig, SizeLossTrainer


def initialize_trainer(model_type, **kwargs):
    """Initialize a trainer for model.

    Args:
        model_type: either 'wesup', 'wesupv2, 'cdws' 'sizeloss' or 'mild'
        kwargs: additional training config

    Returns:
        model: a model instance with given type
    """

    if model_type == 'wesup':
        kwargs = {**WESUPConfig().to_dict(), **kwargs}
        model = WESUP(**kwargs)
        trainer = WESUPTrainer(model, **kwargs)
    elif model_type == 'wesupv2':
        kwargs = {**WESUPV2Config().to_dict(), **kwargs}
        model = WESUPV2(**kwargs)
        trainer = WESUPV2Trainer(model, **kwargs)
    elif model_type == 'cdws':
        kwargs = {**CDWSConfig().to_dict(), **kwargs}
        model = CDWS(**kwargs)
        trainer = CDWSTrainer(model, **kwargs)
    elif model_type == 'sizeloss':
        kwargs = {**SizeLossConfig().to_dict(), **kwargs}
        model = SizeLoss(**kwargs)
        trainer = SizeLossTrainer(model, **kwargs)
    elif model_type == 'mild':
        kwargs = {**MILDNetConfig().to_dict(), **kwargs}
        model = MILDNet()
        trainer = MILDNetTrainer(model, **kwargs)
    else:
        raise ValueError(f'Unsupported model: {model_type}')

    return trainer
