class WessupConfig:
    """Configuration for Wessup model."""

    # Rescale factor to subsample input images.
    rescale_factor = 0.5

    # mult-scale range for training
    multiscale_range = (0.4, 0.6)

    # Number of target classes.
    n_classes = 2

    # Class weights for cross-entropy loss function.
    class_weights = (3, 1)

    # CNN backbone (currently only VGG, ResNet and DenseNet are supported)
    backbone = 'vgg16'

    # Superpixel parameters.
    sp_area = 50
    sp_compactness = 40

    # Weight for label-propagated samples when computing loss function
    propagate_threshold = 0.8

    # Weight for label-propagated samples when computing loss function
    propagate_weight = 0.5

    # Optimization parameters.
    momentum = 0.9
    weight_decay = 0.001

    # Whether to freeze backbone.
    freeze_backbone = False

    epsilon = 1e-7


config = WessupConfig()
