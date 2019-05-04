# Wessup

WEakly Supervised SUPerpixels for medical image segmentation using dot annotation.

## Data Preparation

### MICCAI 2015 Gland Segmentation (GlaS)

[GlaS challenge](https://warwick.ac.uk/fac/sci/dcs/research/tia/glascontest/) is a well-known H&E stained digital pathology dataset for medical image segmentation. Download the dataset from [here](https://warwick.ac.uk/fac/sci/dcs/research/tia/glascontest/download/warwick_qu_dataset_released_2016_07_08.zip). Then run the following convenience script to organize the dataset:

```bash
$ python prepare_glas.py /path/to/downloaded/dataset -o data_glas
```

The mask-level fully-annotated dataset `data_glas` looks like this:

```
data_glas
├── train
│   ├── images
│   │   ├── train-1.png
│   │   └── train-2.png
│   └── masks
│       ├── train-1.png
│       └── train-2.png
├── val
│   ├── images
│   │   ├── val-1.png
│   │   └── val-2.png
│   └── masks
│       ├── val-1.png
│       └── val-2.png
└── val-whole
    ├── images
    │   ├── val-1.png
    │   └── val-2.png
    └── masks
        ├── val-1.png
        └── val-2.png
```

Note that:

- `train` directory contains original images and masks
- `val` directory contains cropped patches to be validated
- `val-whole` directory contains whole images and masks to be validated

#### Generating point labels

```bash
$ python scripts/generate_dot_annotation.py data_glas -p 1e-4
```

> The `-p` or `--label-percent` argument is for controlling the percentage of labeled pixels. Larger value means stronger supervision.

Then `labels` directory storing point labels will be generated alongside `images` and `masks`. Each csv file within `labels` directory correspond to a training image, with each row (a triple) representing a point:

```csv
p1_top,p1_left,p1_class
p2_top,p2_left,p2_class
```

#### Visualizing point labels

```bash
$ python scripts/visualize_dot_annotation.py data_glas/train
```

You will see visualization outputs in `data_glas/train/viz`.

## Training

### Training from scratch

```bash
$ python train.py /path/to/dataset -b resnet50 -w 5 -e 50 -j 4
```

Notes on important arguments:

- `-b` or `--backbone` takes a string representing the CNN backbone, such as `vgg13` or `resnet50`. Currently, only VGG Family (`vgg11`, `vgg13`, `vgg16` and `vgg19`), ResNet family (`resnet18`, `resnet34`, `resnet50`, `resnet101` and `resnet152`) and DenseNet family (`densenet121`, `densenet161`, `densenet169` and `densenet201`) are supported.
- `-w` or `--warmup` takes an integer, which is the number of warmup epochs where only parameters of the MLP classifier is updated.
- `-e` or `--epochs` takes an integer, which is the number of training epochs
- `-j` or `--jobs` is the number of workers for data preprocessing and loading. Since SLIC operation can take nonnegligible amount of time, more workers can bring about significant speedup for training

For all arguments and options, run `python train.py -h`.

### Resume training from a checkpoint

```bash
$ python train.py /path/to/dataset -e 20 -j 4 -r /path/to/checkpoint
```

### Recording multple runs

By default, each run will be stored within a timestamped directory within `records`. The structure of a record directory is as follows:

```
records/20190423-1122-AM
├── checkpoints
│   ├── ckpt.0001.pth
│   ├── ckpt.0002.pth
│   └── ckpt.0003.pth
├── curves
│   ├── loss.png
│   ├── pixel_acc.png
│   └── sp_acc.png
├── viz
│   ├── metrics.csv
│   ├── train_1.png
│   ├── train_1.pred.png
│   └── train_1.gt.png
├── history.csv
├── params
│   ├── 0.json
│   └── 1.json
└── source
```

- `checkpoints` directory stores all training checkpoints
- `curves` stores learning curves for loss and all metrics
- `params` stores CLI and configuration parameters
- `viz` contains visualization of model predictions and whole image metrics
- `source` stores a snapshot of all source code file
- `history.csv` records the training history

## Inference

```bash
$ python infer.py /path/to/test/data -c /path/to/checkpoint -o prediction -j 4
```

Test images should be placed in a subdirectory named `images`.

### Testing on GlaS dataset

```bash
$ python test.py data_glas -c /path/to/checkpoint
```
