import torch
import time
from models.wesupv2 import WESUPV2

model = WESUPV2(fm_size=(140, 200)).cuda()
input_ = torch.randn(3, 3, 280, 400).cuda()
start = time.time()
h, fmaps = model(input_)
print('forward time', time.time() - start)
print('h.size()', h.size())
print('fmaps.size()', fmaps.size())
