"""
Configuration module.
"""

# Rescale factor to subsample input images.
RESCALE_FACTOR = 0.5

# Number of target classes.
N_CLASSES = 2

# Class weights for cross-entropy loss function
CLASS_WEIGHTS = (2, 1)

# Threshold of similarity for label propagation
PROPAGATE_THRESHOLD = 0.9

# Weight for label-propagated samples when computing loss function
PROPAGATE_WEIGHT = 0.5

# Superpixel parameters.
SP_AREA = 50
SP_COMPACTNESS = 40

# Period (epochs) for saving checkpoints
CHECKPOINT_PERIOD = 10

# Optimization parameters
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005

# Smooth item for numerical stability
EPSILON = 1e-7
