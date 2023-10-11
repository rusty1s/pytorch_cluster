from itertools import product

import pytest
import scipy.spatial
import torch
from torch_cluster import knn, knn_graph
from torch_cluster.testing import devices, grad_dtypes, tensor


def to_set(edge_index):
    return set([(i, j) for i, j in edge_index.t().tolist()])


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

    edge_index = knn(x, y, 2)
    assert to_set(edge_index) == set([(0, 2), (0, 3), (1, 0), (1, 1)])

    jit = torch.jit.script(knn)
    edge_index = jit(x, y, 2)
    assert to_set(edge_index) == set([(0, 2), (0, 3), (1, 0), (1, 1)])

    edge_index = knn(x, y, 2, batch_x, batch_y)
    assert to_set(edge_index) == set([(0, 2), (0, 3), (1, 4), (1, 5)])

    if x.is_cuda:
        edge_index = knn(x, y, 2, batch_x, batch_y, cosine=True)
        assert to_set(edge_index) == set([(0, 2), (0, 3), (1, 4), (1, 5)])

    # Skipping a batch
    batch_x = tensor([0, 0, 0, 0, 2, 2, 2, 2], torch.long, device)
    batch_y = tensor([0, 2], torch.long, device)
    edge_index = knn(x, y, 2, batch_x, batch_y)
    assert to_set(edge_index) == set([(0, 2), (0, 3), (1, 4), (1, 5)])


@pytest.mark.parametrize('dtype,device', product(grad_dtypes, devices))
def test_knn_graph(dtype, device):
    x = tensor([
        [-1, -1],
        [-1, +1],
        [+1, +1],
        [+1, -1],
    ], dtype, device)

    edge_index = knn_graph(x, k=2, flow='target_to_source')
    assert to_set(edge_index) == set([(0, 1), (0, 3), (1, 0), (1, 2), (2, 1),
                                      (2, 3), (3, 0), (3, 2)])

    edge_index = knn_graph(x, k=2, flow='source_to_target')
    assert to_set(edge_index) == set([(1, 0), (3, 0), (0, 1), (2, 1), (1, 2),
                                      (3, 2), (0, 3), (2, 3)])

    jit = torch.jit.script(knn_graph)
    edge_index = jit(x, k=2, flow='source_to_target')
    assert to_set(edge_index) == set([(1, 0), (3, 0), (0, 1), (2, 1), (1, 2),
                                      (3, 2), (0, 3), (2, 3)])


@pytest.mark.parametrize('dtype,device', product([torch.float], devices))
def test_knn_graph_large(dtype, device):
    x = torch.randn(1000, 3, dtype=dtype, device=device)

    edge_index = knn_graph(x, k=5, flow='target_to_source', loop=True)

    tree = scipy.spatial.cKDTree(x.cpu().numpy())
    _, col = tree.query(x.cpu(), k=5)
    truth = set([(i, j) for i, ns in enumerate(col) for j in ns])

    assert to_set(edge_index.cpu()) == truth
