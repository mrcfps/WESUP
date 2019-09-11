import sys
import os.path as osp

sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))

from .cdws_mil import CDWS
from .mild_net import MILDNet
from .wesup import WESUP
from .sizeloss import SizeLoss


def initialize_model(model_type, checkpoint=None):
    """Initialize a model.

    Args:
        model_type: either 'wessup', 'cdws' 'sizeloss' or 'mild'
        checkpoint: model checkpoint
    
    Returns:
        model: a model instance with given type
    """

    if model_type == 'wessup':
        model = WESUP(checkpoint=checkpoint)
    elif model_type == 'cdws':
        model = CDWS(checkpoint=checkpoint)
    elif model_type == 'sizeloss':
        model = SizeLoss(checkpoint=checkpoint)
    elif model_type == 'mild':
        model = MILDNet(checkpoint=checkpoint)
    else:
        raise ValueError(f'Unsupported model: {model_type}')
    
    return model
