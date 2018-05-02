import torch
import cluster_cuda

dtype = torch.float
device = torch.device('cuda')


def grid_cluster(pos, size, start=None, end=None):
    start = pos.t().min(dim=1)[0] if start is None else start
    end = pos.t().max(dim=1)[0] if end is None else end
    return cluster_cuda.grid(pos, size, start, end)


pos = torch.tensor(
    [[1, 1], [3, 3], [5, 5], [7, 7]], dtype=dtype, device=device)
size = torch.tensor([2, 2, 1, 1, 4, 2, 1], dtype=dtype, device=device)
# print('pos', pos.tolist())
# print('size', size.tolist())
cluster = grid_cluster(pos, size)
print('result', cluster.tolist(), cluster.type())
