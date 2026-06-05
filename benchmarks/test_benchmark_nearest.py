import importlib.util
from itertools import product
from typing import Optional

import pytest
import torch
import torch_cluster as tc

nearest = tc.nearest

pytestmark = pytest.mark.skipif(
    not (
        torch.ops.torch_cluster.cuda_version() != -1
        and importlib.util.find_spec('triton') is not None
    ),
    reason='CUDA and Triton are required for Triton benchmark tests.',
)

NEAREST_SIZES = [
    (256, 128),
    (1024, 512),
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
NEAREST_GROUPS = [1, 2, 4, 8, 16, 32]
FEATURES = [8, 64, 200]


def _assert_nearest_within_cuda(
    out_cuda: torch.Tensor,
    out_triton: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    tol: Optional[float] = None,
) -> None:
    if tol is None:
        tol = 5 * torch.finfo(x.dtype).eps
    x_f = x.float()
    y_f = y.float()
    cuda_dist = ((x_f - y_f[out_cuda]) ** 2).sum(dim=1)
    triton_dist = ((x_f - y_f[out_triton]) ** 2).sum(dim=1)
    thresh = cuda_dist + tol
    margin = (triton_dist - thresh).max().item()
    max_diff = torch.abs(triton_dist - cuda_dist).max().item()
    print(f"[nearest][match] max_margin={margin:.6e} tol={tol:.1e}")
    print(f"[nearest][match] max_diff={max_diff:.6e} tol={tol:.1e}")
    assert (triton_dist <= thresh).all()
    assert max_diff <= tol


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


def _nearest_param_grid():
    return (
        (*p[0], p[1], p[2])
        for p in product(NEAREST_SIZES, NEAREST_GROUPS, FEATURES)
        if p[1] <= min(p[0])
    )


@pytest.mark.parametrize(
    'num_x,num_y,num_groups,num_features',
    _nearest_param_grid(),
)
@pytest.mark.benchmark(group="nearest")
def test_triton_nearest_benchmark_cuda(
    benchmark,
    num_x,
    num_y,
    num_groups,
    num_features,
):
    torch.manual_seed(123)
    x = torch.randn(num_x, num_features, device='cuda')
    y = torch.randn(num_y, num_features, device='cuda')
    groups = min(num_groups, x.size(0), y.size(0))
    batch_x = _make_batch(num_x, groups, x.device)
    batch_y = _make_batch(num_y, groups, y.device)

    def cuda_fn():
        nearest(x, y, batch_x, batch_y, use_triton=False)

    for _ in range(5):
        cuda_fn()
        torch.cuda.synchronize()

    benchmark(cuda_fn)
    print(
        f"[nearest][cuda] num_x={num_x} num_y={num_y} groups={groups}"
    )


@pytest.mark.parametrize(
    'num_x,num_y,num_groups,num_features',
    _nearest_param_grid(),
)
@pytest.mark.benchmark(group="nearest")
def test_triton_nearest_benchmark_triton(
    benchmark,
    num_x,
    num_y,
    num_groups,
    num_features,
):
    torch.manual_seed(123)
    x = torch.randn(num_x, num_features, device='cuda')
    y = torch.randn(num_y, num_features, device='cuda')
    groups = min(num_groups, x.size(0), y.size(0))
    batch_x = _make_batch(num_x, groups, x.device)
    batch_y = _make_batch(num_y, groups, y.device)

    def cuda_fn():
        return nearest(x, y, batch_x, batch_y, use_triton=False)

    def triton_fn():
        return nearest(x, y, batch_x, batch_y, use_triton=True)

    for i in range(5):
        if i == 0:
            out_cuda = cuda_fn()
            out_triton = triton_fn()
            _assert_nearest_within_cuda(out_cuda, out_triton, x, y)
        else:
            triton_fn()
        torch.cuda.synchronize()

    benchmark(triton_fn)
    print(
        f"[nearest][triton] num_x={num_x} num_y={num_y} groups={groups}"
    )
