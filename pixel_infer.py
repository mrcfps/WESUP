import argparse
import os
from pathlib import Path

import torch

from tqdm import tqdm
from joblib import Parallel, delayed
from skimage.transform import resize
from skimage.io import imread, imsave

from models.wesup import WESUPPixelInference
from utils.data import SegmentationDataset


def read_image_info(img_path):
    return img_path.name, imread(str(img_path)).shape[:2]


def fuse_prediction_and_save(multiple_preds, info):
    name, target_size = info
    fused_pred = sum(multiple_preds) / len(multiple_preds)
    imsave(output_dir / name, fused_pred.round().astype('uint8') * 255, check_contrast=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_root')
    parser.add_argument('-c', '--checkpoint', required=True)
    parser.add_argument('-o', '--output')
    args = parser.parse_args()

    scales = [0.5]
    executor = Parallel(n_jobs=os.cpu_count())

    data_root = Path(args.data_root).expanduser()
    ckpt_path = Path(args.checkpoint).expanduser()

    print('Reading test image information...')
    infos = executor(delayed(lambda img_path: (img_path.name, imread(str(img_path)).shape[:2]))(img_path)
                     for img_path in tqdm(sorted(list((data_root / 'images').iterdir()))))
    print(f'Total of {len(infos)} images read.')

    device = 'cuda'
    if args.output:
        output_dir = Path(args.output).expanduser()
    else:
        output_dir = ckpt_path.parent.parent / f'results-pixel-{len(scales)}scale' / data_root.name

    if not output_dir.exists():
        os.makedirs(output_dir)

    model = WESUPPixelInference().to(device)
    model.load_state_dict(torch.load(args.checkpoint)['model_state_dict'])

    pred_multi = []

    with torch.no_grad():
        for scale in scales:
            print(f'Making inference ({scale}x) ...')
            dataset = SegmentationDataset(args.data_root, rescale_factor=scale, train=False)
            preds = []

            for (img, mask), (name, target_size) in tqdm(zip(dataset, infos), total=len(dataset)):
                img = img.to(device)
                pred = model(img.unsqueeze(0))
                pred = pred.cpu().numpy()[..., 1]

                pred = resize(pred, target_size)
                preds.append(pred)

            pred_multi.append(preds)

    print(f'Saving final predictions to {output_dir} ...')
    executor(delayed(fuse_prediction_and_save)(multiple_preds, info)
             for multiple_preds, info in tqdm(zip(zip(*pred_multi), infos), total=len(infos)))
