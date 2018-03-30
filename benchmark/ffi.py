import torch
from torch_cluster._ext import ffi

cluster = torch.cuda.LongTensor(5)
pos = torch.cuda.FloatTensor([[1, 1], [3, 3], [1, 1], [5, 5], [3, 3]])
size = torch.cuda.FloatTensor([2, 2])
count = torch.cuda.LongTensor([3, 3])

func = ffi.THCCFloatGrid
print(func)

func(cluster, pos, size, count)
print(cluster)
