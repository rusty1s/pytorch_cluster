import pytest
import torch
from torch_cluster import random_walk
from torch_cluster.testing import devices, has_compiler, tensor


@pytest.mark.parametrize('device', devices)
def test_rw_large(device):
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


@pytest.mark.parametrize('device', devices)
def test_rw_small(device):
    row = tensor([0, 1], torch.long, device)
    col = tensor([1, 0], torch.long, device)
    start = tensor([0, 1, 2], torch.long, device)
    walk_length = 4

    out = random_walk(row, col, start, walk_length, num_nodes=3)
    assert out.tolist() == [[0, 1, 0, 1, 0], [1, 0, 1, 0, 1], [2, 2, 2, 2, 2]]

    if has_compiler():
        jit = torch.compile(random_walk)
        assert torch.equal(
            jit(row, col, start, walk_length, num_nodes=3),
            out,
        )


@pytest.mark.parametrize('device', devices)
def test_rw_large_with_edge_indices(device):
    row = tensor([0, 1, 1, 1, 2, 2, 3, 3, 4, 4], torch.long, device)
    col = tensor([1, 0, 2, 3, 1, 4, 1, 4, 2, 3], torch.long, device)
    start = tensor([0, 1, 2, 3, 4], torch.long, device)
    walk_length = 10

    node_seq, edge_seq = random_walk(
        row,
        col,
        start,
        walk_length,
        return_edge_indices=True,
    )
    assert node_seq[:, 0].tolist() == start.tolist()

    for n in range(start.size(0)):
        cur = start[n].item()
        for i in range(1, walk_length):
            assert node_seq[n, i].item() in col[row == cur].tolist()
            cur = node_seq[n, i].item()

    assert (edge_seq != -1).all()


@pytest.mark.parametrize('device', devices)
def test_rw_small_with_edge_indices(device):
    row = tensor([0, 1], torch.long, device)
    col = tensor([1, 0], torch.long, device)
    start = tensor([0, 1, 2], torch.long, device)
    walk_length = 4

    node_seq, edge_seq = random_walk(
        row,
        col,
        start,
        walk_length,
        num_nodes=3,
        return_edge_indices=True,
    )
    assert node_seq.tolist() == [
        [0, 1, 0, 1, 0],
        [1, 0, 1, 0, 1],
        [2, 2, 2, 2, 2],
    ]
    assert edge_seq.tolist() == [
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, -1, -1, -1],
    ]
