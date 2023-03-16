from itertools import product

import pytest
import torch
from torch_cluster import nearest
from torch_cluster.testing import devices, grad_dtypes, tensor


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

    # Invalid input: instance 1 only in batch_x
    batch_x = tensor([0, 0, 0, 0, 1, 1, 1, 1], torch.long, device)
    batch_y = tensor([0, 0, 0, 0], torch.long, device)
    with pytest.raises(ValueError):
        nearest(x, y, batch_x, batch_y)

    # Invalid input: instance 1 only in batch_x (implicitly as batch_y=None)
    with pytest.raises(ValueError):
        nearest(x, y, batch_x, batch_y=None)

    # Invalid input: instance 2 only in batch_x
    # (i.e.instance in the middle missing)
    batch_x = tensor([0, 0, 1, 1, 2, 2, 3, 3], torch.long, device)
    batch_y = tensor([0, 1, 3, 3], torch.long, device)
    with pytest.raises(ValueError):
        nearest(x, y, batch_x, batch_y)

    # Invalid input: batch_x unsorted
    batch_x = tensor([0, 0, 1, 0, 0, 0, 0], torch.long, device)
    batch_y = tensor([0, 0, 1, 1], torch.long, device)
    with pytest.raises(ValueError):
        nearest(x, y, batch_x, batch_y)

    # Invalid input: batch_y unsorted
    batch_x = tensor([0, 0, 0, 0, 1, 1, 1, 1], torch.long, device)
    batch_y = tensor([0, 0, 1, 0], torch.long, device)
    with pytest.raises(ValueError):
        nearest(x, y, batch_x, batch_y)
