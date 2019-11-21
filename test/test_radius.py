from itertools import product

import pytest
import torch
from torch_cluster import radius, radius_graph

from .utils import grad_dtypes, devices, tensor


def coalesce(index):
    N = index.max().item() + 1
    tensor = torch.sparse_coo_tensor(index, index.new_ones(index.size(1)),
                                     torch.Size([N, N]))
    return tensor.coalesce().indices()


@pytest.mark.parametrize('dtype,device', product(grad_dtypes, devices))
def test_radius(dtype, device):
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
        [0, 0],
        [0, 1],
    ], dtype, device)

    batch_x = tensor([0, 0, 0, 0, 1, 1, 1, 1], torch.long, device)
    batch_y = tensor([0, 1], torch.long, device)

    out = radius(x, y, 2, batch_x, batch_y, max_num_neighbors=4)
    assert coalesce(out).tolist() == [[0, 0, 0, 0, 1, 1], [0, 1, 2, 3, 5, 6]]


@pytest.mark.parametrize('dtype,device', product(grad_dtypes, devices))
def test_radius_graph(dtype, device):
    x = tensor([
        [-1, -1],
        [-1, +1],
        [+1, +1],
        [+1, -1],
    ], dtype, device)

    row, col = radius_graph(x, r=2, flow='target_to_source')
    col = col.view(-1, 2).sort(dim=-1)[0].view(-1)
    assert row.tolist() == [0, 0, 1, 1, 2, 2, 3, 3]
    assert col.tolist() == [1, 3, 0, 2, 1, 3, 0, 2]

    row, col = radius_graph(x, r=2, flow='source_to_target')
    row = row.view(-1, 2).sort(dim=-1)[0].view(-1)
    assert row.tolist() == [1, 3, 0, 2, 1, 3, 0, 2]
    assert col.tolist() == [0, 0, 1, 1, 2, 2, 3, 3]
