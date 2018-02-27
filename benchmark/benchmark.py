import time

import torch
from torch_cluster import sparse_grid_cluster

n = 90000000
s = 1 / 64

print('GPU ===================')

t = time.perf_counter()
pos = torch.cuda.FloatTensor(n, 3).uniform_(0, 1)
size = torch.cuda.FloatTensor([s, s, s])
torch.cuda.synchronize()
print('Init:', time.perf_counter() - t)

t_all = time.perf_counter()
sparse_grid_cluster(pos, size)
torch.cuda.synchronize()
t_all = time.perf_counter() - t_all
print('All:', t_all)

print('CPU ===================')

pos = pos.cpu()
size = size.cpu()

t_all = time.perf_counter()
sparse_grid_cluster(pos, size)
t_all = time.perf_counter() - t_all
print('All:', t_all)
