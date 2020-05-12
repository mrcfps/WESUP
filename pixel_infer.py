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


def main(data_root, checkpoint=None, output_dir=None,
         input_size=None, scales=(0.5,), num_workers=4, device=None):

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if output_dir is None and checkpoint is not None:
        checkpoint = Path(checkpoint)
        output_dir = checkpoint.parent.parent / 'results'
        if not output_dir.exists():
            output_dir.mkdir()

    model = WESUPPixelInference().to(device)
    model.load_state_dict(torch.load(
        checkpoint, map_location=device)['model_state_dict'])

    img_paths = list((data_root / 'images').iterdir())

    with torch.no_grad():
        for img_path in tqdm(img_paths):
            img = TF.to_tensor(Image.open(img_path)).to(device).unsqueeze(0)
            preds = []

            for scale in scales:
                target_size = (int(img.size(2) * scale),
                               int(img.size(3) * scale))
                pred = model(F.interpolate(img, scale_factor=scale,
                                           mode='bilinear', align_corners=True))
                pred = pred[..., 1].unsqueeze(0).unsqueeze(0)
                pred = F.interpolate(pred, size=img.size()[-2:],
                                     align_corners=True, mode='bilinear')
                preds.append(pred.squeeze())

            fused_pred = sum(preds) / len(preds)
            imsave(output_dir / img_path.name.replace('.jpg', '.png'),
                   fused_pred.round().cpu().numpy().astype('uint8') * 255,
                   check_contrast=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_root')
    parser.add_argument('-c', '--checkpoint', required=True)
    parser.add_argument('-s', '--scales', default='0.5')
    parser.add_argument('-o', '--output')
    args = parser.parse_args()

    scales = tuple(float(s) for s in args.scales.split(','))

    data_root = Path(args.data_root).expanduser()
    ckpt_path = Path(args.checkpoint).expanduser()

    if args.output:
        output_dir = Path(args.output).expanduser()
    else:
        output_dir = ckpt_path.parent.parent / \
            f'results-pixel-{args.scales}' / data_root.name

    main(args.data_root, checkpoint=args.checkpoint,
         output_dir=output_dir, scales=scales, device=device)
