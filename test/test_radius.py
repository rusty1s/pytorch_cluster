from itertools import product

import pytest
import torch
import scipy.spatial
from torch_cluster import radius, radius_graph

from .utils import grad_dtypes, devices, tensor


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

    out = radius(x, y, 2, max_num_neighbors=4)
    assert out.tolist() == [[0, 0, 0, 0, 1, 1, 1, 1], [0, 1, 2, 3, 1, 2, 5, 6]]

    out = radius(x, y, 2, batch_x, batch_y, max_num_neighbors=4)
    assert out.tolist() == [[0, 0, 0, 0, 1, 1], [0, 1, 2, 3, 5, 6]]


@pytest.mark.parametrize('dtype,device', product(grad_dtypes, devices))
def test_radius_graph(dtype, device):
    x = tensor([
        [-1, -1],
        [-1, +1],
        [+1, +1],
        [+1, -1],
    ], dtype, device)

    row, col = radius_graph(x, r=2, flow='target_to_source')
    assert row.tolist() == [0, 0, 1, 1, 2, 2, 3, 3]
    assert col.tolist() == [1, 3, 0, 2, 1, 3, 0, 2]

    row, col = radius_graph(x, r=2, flow='source_to_target')
    assert row.tolist() == [1, 3, 0, 2, 1, 3, 0, 2]
    assert col.tolist() == [0, 0, 1, 1, 2, 2, 3, 3]


@pytest.mark.parametrize('dtype,device', product(grad_dtypes, devices))
def test_radius_graph_large(dtype, device):
    x = torch.randn(1000, 3)

    row, col = radius_graph(x, r=0.5, flow='target_to_source', loop=True,
                            max_num_neighbors=1000, num_workers=6)
    pred = set([(i, j) for i, j in zip(row.tolist(), col.tolist())])

    tree = scipy.spatial.cKDTree(x.numpy())
    col = tree.query_ball_point(x.cpu(), r=0.5)
    truth = set([(i, j) for i, ns in enumerate(col) for j in ns])

    assert pred == truth
