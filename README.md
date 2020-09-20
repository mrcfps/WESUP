# WESUP

Source code for our paper *Weakly Supervised Histopathology Image Segmentation with Sparse Point Annotations*.

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
└── val
    ├── images
    │   ├── val-1.png
    │   └── val-2.png
    └── masks
        ├── val-1.png
        └── val-2.png
```

### Colorectal Adenocarcinoma Gland (CRAG) dataset

Download the dataset from this [link](https://warwick.ac.uk/fac/sci/dcs/research/tia/data/mildnet/). Then organize this dataset like GlaS mentioned above.

### Generating point labels

```bash
$ python scripts/generate_points.py /path/to/dataset -p 1e-4
```

> The `-p` or `--label-percent` argument is for controlling the percentage of labeled pixels. Larger value means stronger supervision.

Then `labels` directory storing point labels will be generated alongside `images` and `masks`. Each csv file within `labels` directory correspond to a training image, with each row (a triple) representing a point:

```csv
p1_top,p1_left,p1_class
p2_top,p2_left,p2_class
```

### Visualizing point labels

```bash
$ python scripts/visualize_points.py data_glas/train
```

You will see visualization outputs in `data_glas/train/viz`.

## Training

### Training from scratch

```bash
$ python train.py /path/to/dataset --epochs 100
```

### Resume training from a checkpoint

```bash
$ python train.py /path/to/dataset --epochs 100 --checkpoint /path/to/checkpoint
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

## Inference

We offer four types of inference utilities:

- Superpixel-wise inference (the `infer.py` script)
- Superpixel-wise inference with tiling strategy (the `infer_tile.py` script)
- Pixel-wise inference (the `pixel_infer.py` script)
- Pixel-wise inference with tiling strategy (the `pixel_infer_tile.py` script)

Example:

```bash
$ python infer.py /path/to/test/data --checkpoint /path/to/checkpoint
$ python pixel_infer_tile.py /path/to/test/data --checkpoint /path/to/checkpoint --patch-size 400
```
