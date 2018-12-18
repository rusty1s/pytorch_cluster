from itertools import product

import pytest
import torch
from torch_cluster import nearest

from .utils import grad_dtypes, devices, tensor


@pytest.mark.parametrize('dtype,device', product(grad_dtypes, devices))
def test_nearest(dtype, device):
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

    out = nearest(x, y, batch_x, batch_y)
    assert out.tolist() == [0, 0, 1, 1, 2, 2, 3, 3]

    out = nearest(x, y)
    assert out.tolist() == [0, 0, 1, 1, 2, 2, 3, 3]
