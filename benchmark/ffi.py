import torch
from torch_cluster._ext import ffi

print(ffi.__dict__)

print(ffi.THByteGrid)

cluster = torch.LongTensor(5)
pos = torch.Tensor([[1, 1], [1, 1], [3, 3], [4, 4], [3, 3]])
size = torch.Tensor([2, 2])
count = torch.LongTensor([3, 3])

ffi.THFloatGrid(cluster, pos, size, count)
print(cluster)

cluster = torch.LongTensor(3)
row = torch.LongTensor([0, 0, 1, 1, 2, 2])
col = torch.LongTensor([1, 2, 0, 2, 0, 1])
deg = torch.LongTensor([2, 2, 2])
weight = torch.Tensor([1, 2, 1, 1, 2, 1])

ffi.THFloatGreedy(cluster, row, col, deg, weight)
print(cluster)
