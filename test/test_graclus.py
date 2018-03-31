import torch
from torch_cluster import graclus_cluster


def test_graclus_cluster_cpu():
    row = torch.LongTensor([0, 0, 1, 1, 1, 2, 2, 2, 3, 3])
    col = torch.LongTensor([1, 2, 0, 2, 3, 0, 1, 3, 1, 2])
    weight_ = torch.Tensor([1, 2, 1, 3, 2, 2, 3, 1, 2, 1])
    cluster = graclus_cluster(row, col)
    cluster = graclus_cluster(row, col, weight_)
