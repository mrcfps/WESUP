# Wessup

WEakly Supervised SUPerpixels for medical image segmentation using dot annotation.

## Data Preparation

### MICCAI 2015 Gland Segmentation (GlaS)

[GlaS challenge](https://warwick.ac.uk/fac/sci/dcs/research/tia/glascontest/) is a well-known H&E stained digital pathology dataset for medical image segmentation. Download the dataset from [here](https://warwick.ac.uk/fac/sci/dcs/research/tia/glascontest/download/warwick_qu_dataset_released_2016_07_08.zip). Then run the following convenience script to organize the dataset:

```bash
$ python prepare_glas.py /path/to/downloaded/dataset -o full_anno_glas
```

The mask-level fully-annotated dataset `full_anno_glas` looks like this:

```
full_anno_glas
├── train
│   ├── images
│   │   ├── train-1.png
│   │   └── train-2.png
│   └── masks
│       ├── train-1.png
│       └── train-2.png
└── val
    ├── images
    │   ├── val-1.png
    │   └── val-2.png
    └── masks
        ├── val-1.png
        └── val-2.png
```

To generate weakly-supervised dataset with dot annotation, run the following command:

```bash
$ python scripts/generate_dot_annotation.py full_anno_glas -o dot_anno_glas -p 0.5
```

> The `-p` or `--label-percent` argument is for controlling the percentage of labeled objects. Larger value means stronger supervision.

The generated dataset `full_anno_glas` looks like this:

```
dot_anno_glas
├── train
│   ├── images
│   │   ├── train-1.png
│   │   └── train-2.png
│   └── labels
│       ├── train-1.csv
│       └── train-2.csv
└── val
    ├── images
    │   ├── val-1.png
    │   └── val-2.png
    └── labels
        ├── val-1.csv
        └── val-2.csv
```

Dot annotation are stored in a csv file, with each row (a triple) representing a point:

```csv
p1_top,p1_left,p1_label
p2_top,p2_left,p2_label
```

## Training

### Training from scratch

```bash
$ python train.py /path/to/dataset -w 5 -e 50 -j 4
```

Notes on important arguments:

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
├── history.csv
├── params
│   ├── 0.json
│   └── 1.json
└── source
```

- `checkpoints` directory stores all training checkpoints
- `curves` stores learning curves for loss and all metrics
- `params` stores CLI and configuration parameters
- `source` stores a snapshot of all source code file
- `history.csv` records the training history
