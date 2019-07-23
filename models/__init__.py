import sys
import os.path as osp

sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))

from .cdws_mil import CDWS
from .wessup import Wessup
from .sizeloss import SizeLoss


def initialize_model(model_type, checkpoint=None):
    """Initialize a model.

    Args:
        model_type: either 'wessup', 'cdws' or 'sizeloss'
        checkpoint: model checkpoint
    
    Returns:
        model: a model instance with given type
    """

    if model_type == 'wessup':
        model = Wessup(checkpoint=checkpoint)
    elif model_type == 'cdws':
        model = CDWS(checkpoint=checkpoint)
    elif model_type == 'sizeloss':
        model = SizeLoss(checkpoint=checkpoint)
    else:
        raise ValueError(f'Unsupported model: {model_type}')
    
    return model
