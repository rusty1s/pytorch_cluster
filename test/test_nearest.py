import importlib.util
from itertools import product

import pytest
import torch
from torch_cluster import nearest
from torch_cluster.testing import devices, grad_dtypes, tensor, triton_wrap


@pytest.mark.skipif(
    not (
        torch.cuda.is_available()
        and importlib.util.find_spec('triton') is not None
    ),
    reason='CUDA and Triton are required for Triton parity tests.',
)
def test_nearest_triton_matches_cuda():
    torch.manual_seed(123)
    x = torch.randn(128, 8, device='cuda')
    y = torch.randn(32, 8, device='cuda')
    batch_x = torch.zeros(x.size(0), dtype=torch.long, device='cuda')
    batch_y = torch.zeros(y.size(0), dtype=torch.long, device='cuda')

    out_cuda = nearest(x, y, batch_x, batch_y, use_triton=False)
    out_triton = nearest(x, y, batch_x, batch_y, use_triton=True)
    assert torch.equal(out_cuda, out_triton)

    batch_x = torch.zeros(x.size(0), dtype=torch.long, device='cuda')
    batch_y = torch.zeros(y.size(0), dtype=torch.long, device='cuda')
    batch_x[x.size(0) // 2:] = 1
    batch_y[y.size(0) // 2:] = 1

    out_cuda = nearest(x, y, batch_x, batch_y, use_triton=False)
    out_triton = nearest(x, y, batch_x, batch_y, use_triton=True)
    assert torch.equal(out_cuda, out_triton)


@pytest.mark.parametrize(
    'dtype,device,use_triton',
    triton_wrap(product(grad_dtypes, devices)),
)
def test_nearest(dtype, device, use_triton):
    x = tensor([
        [-1, -1],
        [-1, +1],
        [+1, +1],
        [+1, -1],
        [-2, -2],
        [-2, +2],
        [+2, +2],
        [+2, -2],
    ], dtype, device)
    y = tensor([
        [-1, 0],
        [+1, 0],
        [-2, 0],
        [+2, 0],
    ], dtype, device)

    batch_x = tensor([0, 0, 0, 0, 1, 1, 1, 1], torch.long, device)
    batch_y = tensor([0, 0, 1, 1], torch.long, device)

    out = nearest(x, y, batch_x, batch_y, use_triton=use_triton)
    assert out.tolist() == [0, 0, 1, 1, 2, 2, 3, 3]

    out = nearest(x, y)
    assert out.tolist() == [0, 0, 1, 1, 2, 2, 3, 3]

    # Invalid input: instance 1 only in batch_x
    batch_x = tensor([0, 0, 0, 0, 1, 1, 1, 1], torch.long, device)
    batch_y = tensor([0, 0, 0, 0], torch.long, device)
    with pytest.raises(ValueError):
        nearest(x, y, batch_x, batch_y, use_triton=use_triton)

    # Invalid input: instance 1 only in batch_x (implicitly as batch_y=None)
    with pytest.raises(ValueError):
        nearest(x, y, batch_x, batch_y=None, use_triton=use_triton)

    # Invalid input: instance 2 only in batch_x
    # (i.e.instance in the middle missing)
    batch_x = tensor([0, 0, 1, 1, 2, 2, 3, 3], torch.long, device)
    batch_y = tensor([0, 1, 3, 3], torch.long, device)
    with pytest.raises(ValueError):
        nearest(x, y, batch_x, batch_y, use_triton=use_triton)

    # Invalid input: batch_x unsorted
    batch_x = tensor([0, 0, 1, 0, 0, 0, 0], torch.long, device)
    batch_y = tensor([0, 0, 1, 1], torch.long, device)
    with pytest.raises(ValueError):
        nearest(x, y, batch_x, batch_y, use_triton=use_triton)

    # Invalid input: batch_y unsorted
    batch_x = tensor([0, 0, 0, 0, 1, 1, 1, 1], torch.long, device)
    batch_y = tensor([0, 0, 1, 0], torch.long, device)
    with pytest.raises(ValueError):
        nearest(x, y, batch_x, batch_y, use_triton=use_triton)
