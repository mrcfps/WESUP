from utils.data import SegmentationDataset

dataset = SegmentationDataset('./data_glas_0005/train')
img, segments, adjacency, mask, point_mask = dataset[0]
print(point_mask[..., 0].sum().item())
print(point_mask[..., 1].sum().item())
