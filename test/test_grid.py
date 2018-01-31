import pytest
import torch
from torch_cluster import grid_cluster

from .utils import tensors, Tensor


@pytest.mark.parametrize('tensor', tensors)
def test_grid_cluster_cpu(tensor):
    position = Tensor(tensor, [[0, 0], [9, 9], [2, 8], [2, 2], [8, 3]])
    size = torch.LongTensor([5, 5])
    expected = torch.LongTensor([0, 3, 1, 0, 2])

    output = grid_cluster(position, size)
    assert output.tolist() == expected.tolist()

    output = grid_cluster(position.expand(2, 5, 2), size)
    assert output.tolist() == expected.expand(2, 5).tolist()

    expected = torch.LongTensor([0, 1, 3, 2, 4])
    batch = torch.LongTensor([0, 0, 1, 1, 1])
    output = grid_cluster(position, size, batch)
    assert output.tolist() == expected.tolist()

    output = grid_cluster(position.expand(2, 5, 2), size, batch.expand(2, 5))
    assert output.tolist() == expected.expand(2, 5).tolist()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='no CUDA')
@pytest.mark.parametrize('tensor', tensors)
def test_grid_cluster_gpu(tensor):  # pragma: no cover
    position = Tensor(tensor, [[0, 0], [9, 9], [2, 8], [2, 2], [8, 3]]).cuda()
    size = torch.cuda.LongTensor([5, 5])
    expected = torch.LongTensor([0, 3, 1, 0, 2])

    output = grid_cluster(position, size)
    # assert output.cpu().tolist() == expected.tolist()

    output = grid_cluster(position.expand(2, 5, 2), size)
    # assert output.cpu().tolist() == expected.expand(2, 5).tolist()

    expected = torch.LongTensor([0, 1, 3, 2, 4])
    batch = torch.cuda.LongTensor([0, 0, 1, 1, 1])
    output = grid_cluster(position, size, batch)
    # assert output.cpu().tolist() == expected.tolist()

    output = grid_cluster(position.expand(2, 5, 2), size, batch.expand(2, 5))
    # assert output.cpu().tolist() == expected.expand(2, 5).tolist()
