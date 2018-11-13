from itertools import product

import pytest
import torch
import fps_cuda

from .utils import tensor

dtypes = [torch.float]
devices = [torch.device('cuda')]


@pytest.mark.parametrize('dtype,device', product(dtypes, devices))
def test_fps(dtype, device):
    x = tensor([[-1, -1], [-1, 1], [1, 1], [1, -1]], dtype, device)
    x = x.repeat(2, 1)
    batch = tensor([0, 0, 0, 0, 1, 1, 1, 1], torch.long, device)

    out = fps_cuda.fps(x, batch, 0.5, False)
    print(out)
