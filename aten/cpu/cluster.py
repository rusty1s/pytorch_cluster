import torch

import cluster_cpu


def grid_cluster(pos, size, start=None, end=None):
    start = pos.t().min(dim=1)[0] if start is None else start
    end = pos.t().max(dim=1)[0] if end is None else end
    return cluster_cpu.grid(pos, size, start, end)


def graclus_cluster(row, col, num_nodes):
    return cluster_cpu.graclus(row, col, num_nodes)


# pos = torch.tensor([[1, 1], [3, 3], [5, 5], [7, 7]])
# size = torch.tensor([2, 2])
# start = torch.tensor([0, 0])
# end = torch.tensor([7, 7])
# print('pos', pos.tolist())
# print('size', size.tolist())
# cluster = grid_cluster(pos, size)
# print('result', cluster.tolist(), cluster.dtype)

row = torch.tensor([0, 0, 1, 1, 1, 2, 2, 2, 3, 3])
col = torch.tensor([1, 2, 0, 2, 3, 0, 1, 3, 1, 2])
print(row)
print(col)
print('-----------------')
cluster = graclus_cluster(row, col, 4)
print(cluster)
