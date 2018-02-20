import pytest
import torch
from torch_cluster import grid_cluster

from .utils import tensors, Tensor


@pytest.mark.parametrize('tensor', tensors)
def test_grid_cluster_cpu(tensor):
    position = Tensor(tensor, [2, 6])
    size = torch.LongTensor([5])
    expected = torch.LongTensor([0, 0])
    output, _ = grid_cluster(position, size)
    assert output.tolist() == expected.tolist()

    expected = torch.LongTensor([0, 1])
    output, _ = grid_cluster(position, size, origin=0)
    assert output.tolist() == expected.tolist()

    position = Tensor(tensor, [0, 17, 2, 8, 3])
    expected = torch.LongTensor([0, 2, 0, 1, 0])
    output, _ = grid_cluster(position, size)
    assert output.tolist() == expected.tolist()

    output, _ = grid_cluster(position, size, fake_nodes=True)
    expected = torch.LongTensor([0, 3, 0, 1, 0])
    assert output.tolist() == expected.tolist()

    position = Tensor(tensor, [[0, 0], [9, 9], [2, 8], [2, 2], [8, 3]])
    size = torch.LongTensor([5, 5])
    expected = torch.LongTensor([0, 3, 1, 0, 2])
    output, _ = grid_cluster(position, size)
    assert output.tolist() == expected.tolist()

    position = Tensor(tensor, [[0, 11, 2, 2, 8], [0, 9, 8, 2, 3]]).t()
    output, _ = grid_cluster(position, size)
    assert output.tolist() == expected.tolist()

    output, _ = grid_cluster(position.expand(2, 5, 2), size)
    assert output.tolist() == expected.expand(2, 5).tolist()

    position = position.repeat(2, 1)
    batch = torch.LongTensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    expected = torch.LongTensor([0, 3, 1, 0, 2, 4, 7, 5, 4, 6])
    expected_batch2 = torch.LongTensor([0, 0, 0, 0, 1, 1, 1, 1])
    output, batch2 = grid_cluster(position, size, batch)
    assert output.tolist() == expected.tolist()
    assert batch2.tolist() == expected_batch2.tolist()

    output, C = grid_cluster(position, size, batch, fake_nodes=True)
    expected = torch.LongTensor([0, 5, 1, 0, 2, 6, 11, 7, 6, 8])
    assert output.tolist() == expected.tolist()
    assert C == 12


@pytest.mark.skipif(not torch.cuda.is_available(), reason='no CUDA')
@pytest.mark.parametrize('tensor', tensors)
def test_grid_cluster_gpu(tensor):  # pragma: no cover
    position = Tensor(tensor, [2, 6]).cuda()
    size = torch.cuda.LongTensor([5])
    expected = torch.LongTensor([0, 0])
    output, _ = grid_cluster(position, size)
    assert output.cpu().tolist() == expected.tolist()

    expected = torch.LongTensor([0, 1])
    output, _ = grid_cluster(position, size, origin=0)
    assert output.cpu().tolist() == expected.tolist()

    position = Tensor(tensor, [0, 17, 2, 8, 3]).cuda()
    expected = torch.LongTensor([0, 2, 0, 1, 0])
    output, _ = grid_cluster(position, size)
    assert output.cpu().tolist() == expected.tolist()

    output, _ = grid_cluster(position, size, fake_nodes=True)
    expected = torch.LongTensor([0, 3, 0, 1, 0])
    assert output.cpu().tolist() == expected.tolist()

    position = Tensor(tensor, [[0, 0], [9, 9], [2, 8], [2, 2], [8, 3]])
    position = position.cuda()
    size = torch.cuda.LongTensor([5, 5])
    expected = torch.LongTensor([0, 3, 1, 0, 2])
    output, _ = grid_cluster(position, size)
    assert output.cpu().tolist() == expected.tolist()

    position = Tensor(tensor, [[0, 11, 2, 2, 8], [0, 9, 8, 2, 3]])
    position = position.cuda().t()
    output, _ = grid_cluster(position, size)
    assert output.cpu().tolist() == expected.tolist()

    output, _ = grid_cluster(position.expand(2, 5, 2), size)
    assert output.tolist() == expected.expand(2, 5).tolist()

    position = position.repeat(2, 1)
    batch = torch.cuda.LongTensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    expected = torch.LongTensor([0, 3, 1, 0, 2, 4, 7, 5, 4, 6])
    expected_batch2 = torch.LongTensor([0, 0, 0, 0, 1, 1, 1, 1])
    output, batch2 = grid_cluster(position, size, batch)
    assert output.cpu().tolist() == expected.tolist()
    assert batch2.cpu().tolist() == expected_batch2.tolist()

    output, C = grid_cluster(position, size, batch, fake_nodes=True)
    expected = torch.LongTensor([0, 5, 1, 0, 2, 6, 11, 7, 6, 8])
    assert output.cpu().tolist() == expected.tolist()
    assert C == 12
