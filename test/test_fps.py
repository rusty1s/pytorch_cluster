from itertools import product

import pytest
import torch
from torch_cluster import fps

from .utils import grad_dtypes, devices, tensor


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

    out = fps(x, ratio=0.5, random_start=False)
    assert out.sort()[0].tolist() == [0, 5, 6, 7]
