from itertools import product

import pytest
import scipy.spatial
import torch
from torch_cluster import radius, radius_graph
from torch_cluster.testing import (
    devices,
    floating_dtypes,
    has_compiler,
    tensor,
    triton_wrap,
)


def to_set(edge_index):
    return set([(i, j) for i, j in edge_index.t().tolist()])


def to_degree(edge_index):
    _, counts = torch.unique(edge_index[1], return_counts=True)
    return counts.tolist()


def to_batch(nodes):
    return [int(i / 4) for i in nodes]


@pytest.mark.parametrize(
    'dtype,device,use_triton',
    triton_wrap(product(floating_dtypes, devices)),
)
def test_radius(dtype, device, use_triton):
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

    edge_index = radius(x, y, 2, max_num_neighbors=4, use_triton=use_triton)
    assert to_set(edge_index) == set([(0, 0), (0, 1), (0, 2), (0, 3), (1, 1),
                                      (1, 2), (1, 5), (1, 6)])

    if has_compiler():
        jit = torch.compile(radius)
        edge_index = jit(x, y, 2, max_num_neighbors=4, use_triton=use_triton)
        assert to_set(edge_index) == set([
            (0, 0),
            (0, 1),
            (0, 2),
            (0, 3),
            (1, 1),
            (1, 2),
            (1, 5),
            (1, 6),
        ])

    edge_index = radius(
        x,
        y,
        2,
        batch_x,
        batch_y,
        max_num_neighbors=4,
        use_triton=use_triton,
    )
    assert to_set(edge_index) == set([(0, 0), (0, 1), (0, 2), (0, 3), (1, 5),
                                      (1, 6)])

    # Skipping a batch
    batch_x = tensor([0, 0, 0, 0, 2, 2, 2, 2], torch.long, device)
    batch_y = tensor([0, 2], torch.long, device)
    edge_index = radius(
        x,
        y,
        2,
        batch_x,
        batch_y,
        max_num_neighbors=4,
        use_triton=use_triton,
    )
    assert to_set(edge_index) == set([(0, 0), (0, 1), (0, 2), (0, 3), (1, 5),
                                      (1, 6)])


@pytest.mark.parametrize(
    'dtype,device,use_triton',
    triton_wrap(product(floating_dtypes, devices)),
)
def test_radius_graph(dtype, device, use_triton):
    x = tensor([
        [-1, -1],
        [-1, +1],
        [+1, +1],
        [+1, -1],
    ], dtype, device)

    edge_index = radius_graph(
        x,
        r=2.5,
        flow='target_to_source',
        use_triton=use_triton,
    )
    assert to_set(edge_index) == set([(0, 1), (0, 3), (1, 0), (1, 2), (2, 1),
                                      (2, 3), (3, 0), (3, 2)])

    edge_index = radius_graph(
        x,
        r=2.5,
        flow='source_to_target',
        use_triton=use_triton,
    )
    assert to_set(edge_index) == set([(1, 0), (3, 0), (0, 1), (2, 1), (1, 2),
                                      (3, 2), (0, 3), (2, 3)])

    if has_compiler():
        jit = torch.compile(radius_graph)
        edge_index = jit(
            x,
            r=2.5,
            flow='source_to_target',
            use_triton=use_triton,
        )
        assert to_set(edge_index) == set([
            (1, 0),
            (3, 0),
            (0, 1),
            (2, 1),
            (1, 2),
            (3, 2),
            (0, 3),
            (2, 3),
        ])

    edge_index = radius_graph(
        x,
        r=100,
        flow='source_to_target',
        max_num_neighbors=1,
        use_triton=use_triton,
    )
    assert set(to_degree(edge_index)) == set([1])

    x = tensor([
        [-1, -1],
        [-1, -1],
        [-1, -1],
        [-1, -1],
    ], dtype, device)

    edge_index = radius_graph(
        x,
        r=100,
        flow='source_to_target',
        max_num_neighbors=1,
        use_triton=use_triton,
    )
    assert set(to_degree(edge_index)) == set([1])

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
    batch_x = tensor([0, 0, 0, 0, 1, 1, 1, 1], torch.long, device)

    edge_index = radius_graph(
        x,
        r=100,
        batch=batch_x,
        flow='source_to_target',
        max_num_neighbors=1,
        use_triton=use_triton,
    )
    assert set(to_degree(edge_index)) == set([1])
    assert to_batch(edge_index[0]) == batch_x.tolist()


@pytest.mark.parametrize(
    'dtype,device,use_triton',
    triton_wrap(product([torch.float], devices)),
)
def test_radius_graph_large(dtype, device, use_triton):
    x = torch.randn(1000, 3, dtype=dtype, device=device)

    edge_index = radius_graph(
        x,
        r=0.5,
        flow='target_to_source',
        loop=True,
        max_num_neighbors=2000,
        use_triton=use_triton,
    )

    tree = scipy.spatial.cKDTree(x.cpu().numpy())
    col = tree.query_ball_point(x.cpu(), r=0.5)
    truth = set([(i, j) for i, ns in enumerate(col) for j in ns])

    assert to_set(edge_index.cpu()) == truth
