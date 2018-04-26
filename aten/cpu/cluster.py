import torch

import cluster_cpu


def grid_cluster(pos, size, start, end):
    return cluster_cpu.grid(pos, size, start, end)


pos = torch.tensor([[1, 1], [3, 3], [5, 5], [7, 7]], dtype=torch.uint8)
size = torch.tensor([2, 2])
start = torch.tensor([0, 0])
end = torch.tensor([7, 7])
print('pos', pos.tolist())
print('size', size.tolist())
cluster = grid_cluster(pos, size, start, end)
print('result', cluster.tolist(), cluster.dtype)
