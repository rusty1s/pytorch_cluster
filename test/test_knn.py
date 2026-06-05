import importlib.util
from itertools import product

import pytest
import scipy.spatial
import torch
from torch_cluster import knn, knn_graph
from torch_cluster.testing import (
    devices,
    grad_dtypes,
    has_compiler,
    tensor,
    triton_wrap,
)

HAS_CUDA = torch.cuda.is_available()
HAS_TRITON = importlib.util.find_spec('triton') is not None

torch._dynamo.config.capture_scalar_outputs = True


def to_set(edge_index):
    return set([(i, j) for i, j in edge_index.t().tolist()])


@pytest.mark.parametrize(
    'dtype,device,use_triton',
    triton_wrap(product(grad_dtypes, devices)),
)
def test_knn(dtype, device, use_triton):
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

    edge_index = knn(x, y, 2, use_triton=use_triton)
    assert to_set(edge_index) == {(0, 2), (0, 3), (1, 0), (1, 1)}

    if has_compiler():
        jit = torch.compile(knn)
        edge_index = jit(x, y, 2, use_triton=use_triton)
        assert to_set(edge_index) == {(0, 2), (0, 3), (1, 0), (1, 1)}

    edge_index = knn(x, y, 2, batch_x, batch_y, use_triton=use_triton)
    assert to_set(edge_index) == {(0, 2), (0, 3), (1, 4), (1, 5)}

    if x.is_cuda:
        edge_index = knn(
            x,
            y,
            2,
            batch_x,
            batch_y,
            cosine=True,
            use_triton=use_triton,
        )
        assert to_set(edge_index) == {(0, 2), (0, 3), (1, 4), (1, 5)}

    # Skipping a batch
    batch_x = tensor([0, 0, 0, 0, 2, 2, 2, 2], torch.long, device)
    batch_y = tensor([0, 2], torch.long, device)
    edge_index = knn(x, y, 2, batch_x, batch_y, use_triton=use_triton)
    assert to_set(edge_index) == {(0, 2), (0, 3), (1, 4), (1, 5)}


@pytest.mark.parametrize(
    'dtype,device,use_triton',
    triton_wrap(product(grad_dtypes, devices)),
)
def test_knn_graph(dtype, device, use_triton):
    x = tensor([
        [-1, -1],
        [-1, +1],
        [+1, +1],
        [+1, -1],
    ], dtype, device)

    edge_index = knn_graph(
        x,
        k=2,
        flow='target_to_source',
        use_triton=use_triton,
    )
    assert to_set(edge_index) == {
        (0, 1),
        (0, 3),
        (1, 0),
        (1, 2),
        (2, 1),
        (2, 3),
        (3, 0),
        (3, 2),
    }

    edge_index = knn_graph(
        x,
        k=2,
        flow='source_to_target',
        use_triton=use_triton,
    )
    assert to_set(edge_index) == {
        (1, 0),
        (3, 0),
        (0, 1),
        (2, 1),
        (1, 2),
        (3, 2),
        (0, 3),
        (2, 3),
    }

    if has_compiler():
        jit = torch.compile(knn_graph)
        edge_index = jit(x, k=2, flow='source_to_target')
        assert to_set(edge_index) == {
            (1, 0),
            (3, 0),
            (0, 1),
            (2, 1),
            (1, 2),
            (3, 2),
            (0, 3),
            (2, 3),
        }


@pytest.mark.parametrize(
    'dtype,device,use_triton',
    triton_wrap(product([torch.float], devices)),
)
def test_knn_graph_large(dtype, device, use_triton):
    torch.manual_seed(29)
    x = torch.randn(1000, 3, dtype=dtype, device=device)

    edge_index = knn_graph(
        x,
        k=5,
        flow='target_to_source',
        loop=True,
        use_triton=use_triton,
    )

    tree = scipy.spatial.cKDTree(x.cpu().numpy())
    _, col = tree.query(x.cpu(), k=5)
    truth = set([(i, j) for i, ns in enumerate(col) for j in ns])

    assert to_set(edge_index.cpu()) == truth


@pytest.mark.skipif(
    not (HAS_CUDA and HAS_TRITON),
    reason='CUDA and Triton are required for Triton parity tests.',
)
def test_knn_triton_matches_cuda():
    torch.manual_seed(42)
    x = torch.randn(128, 16, device='cuda')
    y = torch.randn(64, 16, device='cuda')
    batch_x = torch.zeros(x.size(0), dtype=torch.long, device='cuda')
    batch_y = torch.zeros(y.size(0), dtype=torch.long, device='cuda')

    out_cuda = knn(
        x,
        y,
        k=8,
        batch_x=batch_x,
        batch_y=batch_y,
        use_triton=False,
    )
    out_triton = knn(
        x,
        y,
        k=8,
        batch_x=batch_x,
        batch_y=batch_y,
        use_triton=True,
    )
    assert to_set(out_cuda) == to_set(out_triton)

    out_cuda = knn(
        x,
        y,
        k=8,
        batch_x=batch_x,
        batch_y=batch_y,
        cosine=True,
        use_triton=False,
    )
    out_triton = knn(
        x,
        y,
        k=8,
        batch_x=batch_x,
        batch_y=batch_y,
        cosine=True,
        use_triton=True,
    )
    assert to_set(out_cuda) == to_set(out_triton)


@pytest.mark.skipif(
    not (HAS_CUDA and HAS_TRITON),
    reason='CUDA and Triton are required for Triton parity tests.',
)
def test_knn_graph_triton_matches_cuda():
    torch.manual_seed(1)
    x = torch.randn(64, 8, device='cuda')
    batch = torch.zeros(x.size(0), dtype=torch.long, device='cuda')

    out_cuda = knn_graph(x, k=4, batch=batch, loop=False, use_triton=False)
    out_triton = knn_graph(x, k=4, batch=batch, loop=False, use_triton=True)
    assert to_set(out_cuda) == to_set(out_triton)
