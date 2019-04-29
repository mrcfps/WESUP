"""
Configuration module.
"""

# Patch size for prediction (same as input size to CNN)
PATCH_SIZE = 384

# Number of target classes.
N_CLASSES = 2

# Class weights for cross-entropy loss function (background, class-1, class-2)
CLASS_WEIGHTS = (1, 2)

# Weight for label-propagated samples when computing loss function
PROPAGATE_WEIGHT = 0.5

# Superpixel parameters.
SP_AREA = 100
SP_COMPACTNESS = 40

# Period (epochs) for saving checkpoints
CHECKPOINT_PERIOD = 5

# Optimization parameters
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0001

INFER_STRIDE = 200

# Period for testing on whole images
WHOLE_IMAGE_TEST_PERIOD = 5

# Smooth item for numerical stability
EPSILON = 1e-7
