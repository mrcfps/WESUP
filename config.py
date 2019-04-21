# Device for computation ('cpu' or 'cuda')
DEVICE = 'cpu'

PATCH_SIZE = 256

# Number of target classes.
N_CLASSES = 3

# Class weights for cross-entropy loss function (background, class-1, class-2)
CLASS_WEIGHTS = (1, 2, 2)

# SLIC parameters.
SLIC_N_SEGMENTS = 500
SLIC_COMPACTNESS = 40

# Period (epochs) for saving checkpoints
CHECKPOINT_PERIOD = 5

# Optimization parameters
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0001
