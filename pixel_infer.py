import argparse
import os
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from tqdm import tqdm
from PIL import Image
from skimage.io import imread, imsave

from models.wesup import WESUPPixelInference


def read_image_info(img_path):
    return img_path.name, imread(str(img_path)).shape[:2]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_root')
    parser.add_argument('-c', '--checkpoint', required=True)
    parser.add_argument('-o', '--output')
    args = parser.parse_args()

    scales = [0.5]

    data_root = Path(args.data_root).expanduser()
    ckpt_path = Path(args.checkpoint).expanduser()

    device = 'cuda'
    if args.output:
        output_dir = Path(args.output).expanduser()
    else:
        output_dir = ckpt_path.parent.parent / f'results-pixel-{len(scales)}scale' / data_root.name

    if not output_dir.exists():
        os.makedirs(output_dir)

    model = WESUPPixelInference().to(device)
    model.load_state_dict(torch.load(args.checkpoint)['model_state_dict'])

    img_paths = list((data_root / 'images').iterdir())

    print('Making inference ...')
    with torch.no_grad():
        for img_path in tqdm(img_paths):
            img = TF.to_tensor(Image.open(img_path)).to(device).unsqueeze(0)
            preds = []

            for scale in scales:
                target_size = (int(img.size(2) * scale), int(img.size(3) * scale))
                pred = model(F.interpolate(img, scale_factor=scale, mode='bilinear'))
                pred = pred[..., 1].unsqueeze(0).unsqueeze(0)
                pred = F.interpolate(pred, size=img.size()[-2:], mode='bilinear')
                preds.append(pred.squeeze())

            fused_pred = sum(preds) / len(preds)
            imsave(output_dir / img_path.name,
                   fused_pred.round().cpu().numpy().astype('uint8') * 255,
                   check_contrast=False)
