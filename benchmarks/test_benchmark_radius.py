import importlib.util
from itertools import product

import pytest
import torch
import torch_cluster as tc

radius = tc.radius
radius_graph = tc.radius_graph

pytestmark = pytest.mark.skipif(
    not (
        torch.ops.torch_cluster.cuda_version() != -1
        and importlib.util.find_spec('triton') is not None
    ),
    reason='CUDA and Triton are required for Triton benchmark tests.',
)

RADIUS_SIZES = [
    (256, 128),
    (512, 256),
    (1024, 512),
    (2048, 1024),
    (4096, 2048),
    (8192, 4096),
    (8192, 8192),
    (8201, 4103),
    (32000, 32000),
    (255, 127),
    (256, 5),
    (1024, 5),
    (4096, 5),
    (255, 5),
]
RADIUS_GROUPS = [1, 2, 4, 8, 16, 32]
FEATURES = [3, 8, 64, 200]


def to_set(edge_index):
    return set([(i, j) for i, j in edge_index.t().tolist()])


def _assert_radius_within_cuda(
    edge_index,
    x,
    y,
    r,
    max_num_neighbors,
    ignore_same_index,
    check_max_neighbors=True,
    tol=None,
):
    if edge_index.numel() == 0:
        return
    if tol is None:
        tol = 5 * torch.finfo(x.dtype).eps
    row, col = edge_index
    x_f = x.float()
    y_f = y.float()
    diffs = x_f[col] - y_f[row]
    dist = (diffs * diffs).sum(dim=1)
    r2 = float(r) * float(r)
    assert (dist <= (r2 + tol)).all()
    if ignore_same_index:
        assert (row != col).all()
    if check_max_neighbors:
        counts = torch.bincount(row, minlength=y.size(0))
        assert (counts <= max_num_neighbors).all()


def _make_batch(
    num_nodes: int,
    num_groups: int,
    device: torch.device,
) -> torch.Tensor:
    groups = max(1, min(num_groups, num_nodes))
    counts = torch.full(
        (groups,),
        num_nodes // groups,
        device=device,
        dtype=torch.long,
    )
    remainder = num_nodes % groups
    if remainder:
        counts[:remainder] += 1
    return torch.repeat_interleave(
        torch.arange(groups, device=device),
        counts,
    )


def _radius_param_grid():
    return (
        (*p[0], p[1], p[2])
        for p in product(RADIUS_SIZES, RADIUS_GROUPS, FEATURES)
        if p[1] <= min(p[0])
    )


@pytest.mark.parametrize(
    'num_x,num_y,num_groups,num_features',
    _radius_param_grid(),
)
@pytest.mark.benchmark(group="radius")
def test_triton_radius_benchmark_cuda(
    benchmark,
    num_x,
    num_y,
    num_groups,
    num_features,
):
    torch.manual_seed(99)
    x = torch.randn(num_x, num_features, device='cuda')
    y = torch.randn(num_y, num_features, device='cuda')
    groups = min(num_groups, x.size(0), y.size(0))
    batch_x = _make_batch(num_x, groups, x.device)
    batch_y = _make_batch(num_y, groups, y.device)

    def cuda_fn():
        radius(
            x,
            y,
            r=1.5,
            batch_x=batch_x,
            batch_y=batch_y,
            max_num_neighbors=32,
            use_triton=False,
        )

    for _ in range(5):
        cuda_fn()
        torch.cuda.synchronize()

    benchmark(cuda_fn)
    print(
        f"[radius][cuda] num_x={num_x} num_y={num_y} groups={groups}"
    )


@pytest.mark.parametrize(
    'num_x,num_y,num_groups,num_features',
    _radius_param_grid(),
)
@pytest.mark.benchmark(group="radius")
def test_triton_radius_benchmark_triton(
    benchmark,
    num_x,
    num_y,
    num_groups,
    num_features,
):
    torch.manual_seed(99)
    x = torch.randn(num_x, num_features, device='cuda')
    y = torch.randn(num_y, num_features, device='cuda')
    groups = min(num_groups, x.size(0), y.size(0))
    batch_x = _make_batch(num_x, groups, x.device)
    batch_y = _make_batch(num_y, groups, y.device)

    def cuda_fn():
        return radius(
            x,
            y,
            r=1.5,
            batch_x=batch_x,
            batch_y=batch_y,
            max_num_neighbors=32,
            use_triton=False,
        )

    def triton_fn():
        return radius(
            x,
            y,
            r=1.5,
            batch_x=batch_x,
            batch_y=batch_y,
            max_num_neighbors=32,
            use_triton=True,
        )

    for i in range(5):
        if i == 0:
            out_triton = triton_fn()
            _assert_radius_within_cuda(
                out_triton,
                x,
                y,
                r=1.5,
                max_num_neighbors=32,
                ignore_same_index=False,
            )
        else:
            triton_fn()
        torch.cuda.synchronize()

    benchmark(triton_fn)
    print(
        f"[radius][triton] num_x={num_x} num_y={num_y} groups={groups}"
    )


@pytest.mark.parametrize('num_x', [256, 1024, 4096, 8192, 255])
@pytest.mark.parametrize('num_groups', [1, 2, 4, 6, 8, 16, 24, 32])
@pytest.mark.benchmark(group="radius_graph")
def test_triton_radius_graph_benchmark_cuda(benchmark, num_x, num_groups):
    torch.manual_seed(199)
    x = torch.randn(num_x, 8, device='cuda')
    groups = min(num_groups, x.size(0))
    batch = _make_batch(num_x, groups, x.device)

    def cuda_fn():
        radius_graph(
            x,
            r=1.5,
            batch=batch,
            loop=False,
            max_num_neighbors=32,
            use_triton=False,
        )

    for _ in range(5):
        cuda_fn()
        torch.cuda.synchronize()

    benchmark(cuda_fn)
    print(
        f"[radius_graph][cuda] num_x={num_x} groups={groups}"
    )


@pytest.mark.parametrize('num_x', [256, 1024, 4096, 8192, 255])
@pytest.mark.parametrize('num_groups', [1, 2, 4, 6, 8, 16, 24, 32])
@pytest.mark.benchmark(group="radius_graph")
def test_triton_radius_graph_benchmark_triton(benchmark, num_x, num_groups):
    torch.manual_seed(199)
    x = torch.randn(num_x, 8, device='cuda')
    groups = min(num_groups, x.size(0))
    batch = _make_batch(num_x, groups, x.device)

    def cuda_fn():
        return radius_graph(
            x,
            r=1.5,
            batch=batch,
            loop=False,
            max_num_neighbors=32,
            use_triton=False,
        )

    def triton_fn():
        return radius_graph(
            x,
            r=1.5,
            batch=batch,
            loop=False,
            max_num_neighbors=32,
            use_triton=True,
        )

    for i in range(5):
        if i == 0:
            out_triton = triton_fn()
            _assert_radius_within_cuda(
                out_triton,
                x,
                x,
                r=1.5,
                max_num_neighbors=32,
                ignore_same_index=True,
                check_max_neighbors=False,
            )
        else:
            triton_fn()
        torch.cuda.synchronize()

    benchmark(triton_fn)
    print(
        f"[radius_graph][triton] num_x={num_x} groups={groups}"
    )
