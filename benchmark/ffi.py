import time
import torch
from torch_cluster._ext import ffi

cluster = torch.cuda.LongTensor(4)
row = torch.cuda.LongTensor([0, 0, 1, 1, 1, 2, 2, 2, 3, 3])
col = torch.cuda.LongTensor([1, 2, 0, 2, 3, 0, 1, 3, 1, 2])
# deg = torch.cuda.LongTensor([2, 3, 3, 2])

func = ffi.THCCGreedy
print(func)

a = 0
torch.cuda.synchronize()
t = time.perf_counter()
# for i in range(100):
func(cluster, row, col)
# a += cluster.sum() / cluster.size(0)
torch.cuda.synchronize()
print(time.perf_counter() - t)
print(cluster)
