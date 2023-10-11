from itertools import product

import pytest
import scipy.spatial
import torch
from torch_cluster import radius, radius_graph
from torch_cluster.testing import devices, grad_dtypes, tensor


def to_set(edge_index):
    return set([(i, j) for i, j in edge_index.t().tolist()])


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

    edge_index = radius(x, y, 2, max_num_neighbors=4)
    assert to_set(edge_index) == set([(0, 0), (0, 1), (0, 2), (0, 3), (1, 1),
                                      (1, 2), (1, 5), (1, 6)])

    jit = torch.jit.script(radius)
    edge_index = jit(x, y, 2, max_num_neighbors=4)
    assert to_set(edge_index) == set([(0, 0), (0, 1), (0, 2), (0, 3), (1, 1),
                                      (1, 2), (1, 5), (1, 6)])

    edge_index = radius(x, y, 2, batch_x, batch_y, max_num_neighbors=4)
    assert to_set(edge_index) == set([(0, 0), (0, 1), (0, 2), (0, 3), (1, 5),
                                      (1, 6)])

    # Skipping a batch
    batch_x = tensor([0, 0, 0, 0, 2, 2, 2, 2], torch.long, device)
    batch_y = tensor([0, 2], torch.long, device)
    edge_index = radius(x, y, 2, batch_x, batch_y, max_num_neighbors=4)
    assert to_set(edge_index) == set([(0, 0), (0, 1), (0, 2), (0, 3), (1, 5),
                                      (1, 6)])


@pytest.mark.parametrize('dtype,device', product(grad_dtypes, devices))
def test_radius_graph(dtype, device):
    x = tensor([
        [-1, -1],
        [-1, +1],
        [+1, +1],
        [+1, -1],
    ], dtype, device)

    edge_index = radius_graph(x, r=2.5, flow='target_to_source')
    assert to_set(edge_index) == set([(0, 1), (0, 3), (1, 0), (1, 2), (2, 1),
                                      (2, 3), (3, 0), (3, 2)])

    edge_index = radius_graph(x, r=2.5, flow='source_to_target')
    assert to_set(edge_index) == set([(1, 0), (3, 0), (0, 1), (2, 1), (1, 2),
                                      (3, 2), (0, 3), (2, 3)])

    jit = torch.jit.script(radius_graph)
    edge_index = jit(x, r=2.5, flow='source_to_target')
    assert to_set(edge_index) == set([(1, 0), (3, 0), (0, 1), (2, 1), (1, 2),
                                      (3, 2), (0, 3), (2, 3)])


@pytest.mark.parametrize('dtype,device', product([torch.float], devices))
def test_radius_graph_large(dtype, device):
    x = torch.randn(1000, 3, dtype=dtype, device=device)

    edge_index = radius_graph(x,
                              r=0.5,
                              flow='target_to_source',
                              loop=True,
                              max_num_neighbors=2000)

    tree = scipy.spatial.cKDTree(x.cpu().numpy())
    col = tree.query_ball_point(x.cpu(), r=0.5)
    truth = set([(i, j) for i, ns in enumerate(col) for j in ns])

    assert to_set(edge_index.cpu()) == truth
