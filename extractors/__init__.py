import os
import sys

from .vgg16 import VGG16FeatureExtractor

# Make sure modules in project root are visible.
sys.path.append(os.path.abspath(os.path.join(__file__, os.pardir, os.pardir)))
