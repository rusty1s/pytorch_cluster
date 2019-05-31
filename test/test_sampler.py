import torch

from torch_cluster import neighbor_sampler


def test_neighbor_sampler():
    torch.manual_seed(1234)

    start = torch.tensor([0, 1])
    cumdeg = torch.tensor([0, 3, 7])

    e_id = neighbor_sampler(start, cumdeg, size=1.0)
    assert e_id.tolist() == [0, 2, 1, 5, 6, 3, 4]

    e_id = neighbor_sampler(start, cumdeg, size=3)
    assert e_id.tolist() == [1, 0, 2, 4, 5, 6]
