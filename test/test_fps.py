from itertools import product

import pytest
import torch
from torch_cluster import fps

from .utils import tensor, grad_dtypes

devices = [torch.device('cuda')]


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA not available')
@pytest.mark.parametrize('dtype,device', product(grad_dtypes, devices))
def test_fps(dtype, device):
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
    batch = tensor([0, 0, 0, 0, 1, 1, 1, 1], torch.long, device)

    out = fps(x, batch, ratio=0.5, random_start=False)
    assert out.tolist() == [0, 2, 4, 6]


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA not available')
@pytest.mark.parametrize('dtype,device', product(grad_dtypes, devices))
def test_fps_speed(dtype, device):
    return
    batch_size, num_nodes = 100, 10000
    x = torch.randn((batch_size * num_nodes, 3), dtype=dtype, device=device)
    batch = torch.arange(batch_size, dtype=torch.long, device=device)
    batch = batch.view(-1, 1).repeat(1, num_nodes).view(-1)

    out = fps(x, batch, ratio=0.5, random_start=True)
    assert out.size(0) == batch_size * num_nodes * 0.5
    assert out.min().item() >= 0 and out.max().item() < batch_size * num_nodes

    batch_size, num_nodes, dim = 100, 300, 128
    x = torch.randn((batch_size * num_nodes, dim), dtype=dtype, device=device)
    batch = torch.arange(batch_size, dtype=torch.long, device=device)
    batch = batch.view(-1, 1).repeat(1, num_nodes).view(-1)
    out = fps(x, batch, ratio=0.5, random_start=True)
    assert out.size(0) == batch_size * num_nodes * 0.5
    assert out.min().item() >= 0 and out.max().item() < batch_size * num_nodes
