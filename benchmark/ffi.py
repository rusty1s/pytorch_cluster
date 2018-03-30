import torch
from torch_cluster._ext import ffi

cluster = torch.cuda.LongTensor(5)
pos = torch.cuda.FloatTensor(5, 2)
size = torch.cuda.FloatTensor(2)
count = torch.cuda.LongTensor(2)

func = ffi.THCCFloatGrid
print(func)

func(cluster, pos, size, count)
