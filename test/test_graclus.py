import pytest
import torch
from torch_cluster import graclus_cluster


def test_graclus():
    edge_index = torch.LongTensor([[0, 0, 0, 1, 2, 3, 3, 3, 4, 5, 5, 5, 6, 6],
                                   [2, 3, 6, 5, 0, 0, 4, 5, 3, 1, 3, 6, 0, 3]])
    edge_attr = torch.Tensor([2, 2, 2, 1, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2])

    graclus_cluster(edge_index, edge_attr)
