import torch

import cluster_cpu
import cluster_cuda


def grid(pos, size, start=None, end=None):
    lib = cluster_cuda if pos.is_cuda else cluster_cpu
    start = pos.t().min(dim=1)[0] if start is None else start
    end = pos.t().max(dim=1)[0] if end is None else end
    return lib.grid(pos, size, start, end)


def graclus(row, col, num_nodes):
    lib = cluster_cuda if pos.is_cuda else cluster_cpu
    return lib.graclus(row, col, num_nodes)


device = torch.device('cuda')
pos = torch.tensor([[1, 1], [3, 3], [5, 5], [7, 7]], device=device)
size = torch.tensor([2, 2], device=device)
print('pos', pos.tolist())
print('size', size.tolist())
cluster = grid(pos, size)
print('result', cluster.tolist(), cluster.dtype, cluster.device)

print('-----------------')

row = torch.tensor([0, 0, 1, 1, 1, 2, 2, 2, 3, 3], device=device)
col = torch.tensor([1, 2, 0, 2, 3, 0, 1, 3, 1, 2], device=device)
print('row', row.tolist())
print('col', col.tolist())
cluster = graclus(row, col, 4)
print('result', cluster.tolist(), cluster.dtype, cluster.device)
