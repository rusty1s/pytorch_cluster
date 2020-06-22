from itertools import product

import pytest
import torch
import scipy.spatial
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

    row, col = knn(x, y, 2)
    assert row.tolist() == [0, 0, 1, 1]
    assert col.tolist() == [2, 3, 0, 1]

    row, col = knn(x, y, 2, batch_x, batch_y)
    assert row.tolist() == [0, 0, 1, 1]
    assert col.tolist() == [2, 3, 4, 5]

    if x.is_cuda:
        row, col = knn(x, y, 2, batch_x, batch_y, cosine=True)
        assert row.tolist() == [0, 0, 1, 1]
        assert col.tolist() == [0, 1, 4, 5]


@pytest.mark.parametrize('dtype,device', product(grad_dtypes, devices))
def test_knn_graph(dtype, device):
    x = tensor([
        [-1, -1],
        [-1, +1],
        [+1, +1],
        [+1, -1],
    ], dtype, device)

    row, col = knn_graph(x, k=2, flow='target_to_source')
    assert row.tolist() == [0, 0, 1, 1, 2, 2, 3, 3]
    assert col.tolist() == [1, 3, 0, 2, 1, 3, 0, 2]

    row, col = knn_graph(x, k=2, flow='source_to_target')
    assert row.tolist() == [1, 3, 0, 2, 1, 3, 0, 2]
    assert col.tolist() == [0, 0, 1, 1, 2, 2, 3, 3]


@pytest.mark.parametrize('dtype,device', product(grad_dtypes, devices))
def test_knn_graph_large(dtype, device):
    x = torch.randn(1000, 3)

    row, col = knn_graph(x, k=5, flow='target_to_source', loop=True,
                         num_workers=6)
    pred = set([(i, j) for i, j in zip(row.tolist(), col.tolist())])

    tree = scipy.spatial.cKDTree(x.numpy())
    _, col = tree.query(x.cpu(), k=5)
    truth = set([(i, j) for i, ns in enumerate(col) for j in ns])

    assert pred == truth
