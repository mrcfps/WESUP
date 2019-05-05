import argparse
import pandas as pd


def report(csv_path):
    df = pd.read_csv(csv_path)
    print('Max pixel_acc', df['pixel_acc'].max())
    print('Max dice', df['dice'].max())
    print('Max val_pixel_acc', df['val_pixel_acc'].max())
    print('Max val_dice', df['val_dice'].max())

    if 'labeled_sp_ratio' in df:
        print('Mean sp_ratio', df['labeled_sp_ratio'].mean())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('csv_path', help='Path to history csv file')
    args = parser.parse_args()

    report(args.csv_path)
