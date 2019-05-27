from utils.data import SegmentationDataset

dataset = SegmentationDataset('./data_glas_0005/train')
img, segments, adjacency, mask, point_mask = dataset[0]
print(point_mask.sum().item())
