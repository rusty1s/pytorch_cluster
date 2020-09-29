import pytest
import torch
from torch_cluster import random_walk

from .utils import devices, tensor


@pytest.mark.parametrize('device', devices)
def test_rw(device):
    row = tensor([0, 1, 1, 1, 2, 2, 3, 3, 4, 4], torch.long, device)
    col = tensor([1, 0, 2, 3, 1, 4, 1, 4, 2, 3], torch.long, device)
    start = tensor([0, 1, 2, 3, 4], torch.long, device)
    walk_length = 10

    out = random_walk(row, col, start, walk_length)
    assert out[:, 0].tolist() == start.tolist()

    for n in range(start.size(0)):
        cur = start[n].item()
        for i in range(1, walk_length):
            assert out[n, i].item() in col[row == cur].tolist()
            cur = out[n, i].item()

    row = tensor([0, 1], torch.long, device)
    col = tensor([1, 0], torch.long, device)
    start = tensor([0, 1, 2], torch.long, device)
    walk_length = 4

    out = random_walk(row, col, start, walk_length, num_nodes=3)
    assert out.tolist() == [[0, 1, 0, 1, 0], [1, 0, 1, 0, 1], [2, 2, 2, 2, 2]]
