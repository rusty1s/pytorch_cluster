from itertools import product

import pytest
import torch
from torch_cluster import knn, knn_graph

from .utils import grad_dtypes, devices, tensor


@pytest.mark.parametrize('dtype,device', product(grad_dtypes, devices))
def test_knn(dtype, device):
    x = tensor([
        [-1, -1],
        [-1, +1],
        [+1, +1],
        [+1, -1],
        [-1, -1],
        [-1, +1],
        [+1, +1],
        [+1, -1],
    ], dtype, device)
    y = tensor([
        [1, 0],
        [-1, 0],
    ], dtype, device)

    batch_x = tensor([0, 0, 0, 0, 1, 1, 1, 1], torch.long, device)
    batch_y = tensor([0, 1], torch.long, device)

    row, col = knn(x, y, 2, batch_x, batch_y)
    col = col.view(-1, 2).sort(dim=-1)[0].view(-1)

    assert row.tolist() == [0, 0, 1, 1]
    assert col.tolist() == [2, 3, 4, 5]


@pytest.mark.parametrize('dtype,device', product(grad_dtypes, devices))
def test_knn_graph(dtype, device):
    x = tensor([
        [-1, -1],
        [-1, +1],
        [+1, +1],
        [+1, -1],
    ], dtype, device)

    row, col = knn_graph(x, k=2)
    col = col.view(-1, 2).sort(dim=-1)[0].view(-1)

    assert row.tolist() == [0, 0, 1, 1, 2, 2, 3, 3]
    assert col.tolist() == [1, 3, 0, 2, 1, 3, 0, 2]
