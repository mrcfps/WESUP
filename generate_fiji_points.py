import csv
import os
from pathlib import Path
from joblib import Parallel, delayed

data_root = Path('~/data/LUSC/train').expanduser()
points_dir = data_root / 'points-0.01'
target_dir = data_root / 'points-fiji'

target_dir.mkdir(exist_ok=True)

# for csv_path in points_dir.glob('*.csv'):
#     with open(str(csv_path)) as fp:
#         points = [[int(d) for d in point] for point in csv.reader(fp)]
#
#     # Write points for the first class
#     content0 = []
#     content1 = []
#
#     for point in points:
#         if point[-1] != 0:
#             content0.append(f'{point[0]} {point[1]}')
#         else:
#             content1.append(f'{point[0]} {point[1]}')
#
#     content0.insert(0, str(len(content0)))
#     content0.insert(0, 'point')
#     content1.insert(0, str(len(content1)))
#     content1.insert(0, 'point')
#
#     with open(target_dir / csv_path.name.replace('.csv', '_0.txt'), 'w') as fp:
#         fp.write('\n'.join(content0))
#
#     with open(target_dir / csv_path.name.replace('.csv', '_1.txt'), 'w') as fp:
#         fp.write('\n'.join(content1))
#
#     print(f'Processed {csv_path}.')


def process(csv_path):
    with open(str(csv_path)) as fp:
        points = [[int(d) for d in point] for point in csv.reader(fp)]

    # Write points for the first class
    content0 = []
    content1 = []

    for point in points:
        if point[-1] != 0:
            content0.append(f'{point[0]} {point[1]}')
        else:
            content1.append(f'{point[0]} {point[1]}')

    content0.insert(0, str(len(content0)))
    content0.insert(0, 'point')
    content1.insert(0, str(len(content1)))
    content1.insert(0, 'point')

    with open(target_dir / csv_path.name.replace('.csv', '_0.txt'), 'w') as fp:
        fp.write('\n'.join(content0))

    with open(target_dir / csv_path.name.replace('.csv', '_1.txt'), 'w') as fp:
        fp.write('\n'.join(content1))

    print(f'Processed {csv_path}.')


executor = Parallel(n_jobs=os.cpu_count())
executor(delayed(process)(csv_path) for csv_path in points_dir.glob('*.csv'))
