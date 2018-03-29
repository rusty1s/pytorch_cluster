import pytest
import torch
from torch_cluster.functions.utils.ffi import ffi_serial, ffi_grid, _get_func


def test_serial_cpu():
    row = torch.LongTensor([0, 0, 1, 1, 1, 2, 2, 2, 3, 3])
    col = torch.LongTensor([1, 2, 0, 2, 3, 0, 1, 3, 1, 2])
    degree = torch.LongTensor([2, 3, 3, 2])
    cluster = ffi_serial(row, col, degree)
    expected_cluster = [0, 0, 2, 2]
    assert cluster.tolist() == expected_cluster

    weight = torch.Tensor([1, 2, 1, 3, 2, 2, 3, 3, 2, 3])
    cluster = ffi_serial(row, col, degree, weight)
    expected_cluster = [0, 1, 0, 1]
    assert cluster.tolist() == expected_cluster


def test_grid_cpu():
    position = torch.Tensor([[0, 0], [11, 9], [2, 8], [2, 2], [8, 3]])
    size = torch.Tensor([5, 5])
    count = torch.LongTensor([3, 2])
    cluster = ffi_grid(position, size, count)
    expected_cluster = [0, 5, 1, 0, 2]
    assert cluster.tolist() == expected_cluster


@pytest.mark.skipif(not torch.cuda.is_available(), reason='no CUDA')
def test_assign_color_gpu():
    output = torch.cuda.LongTensor(60000).fill_(-1)
    func = _get_func('serial', output)
    func(output, output, output, output)
    print((output + 2).sum() / output.size(0))
    print((output + 2)[:10])

    # print(torch.initial_seed())
    # torch.cuda.manual_seed(2)
    # bla = torch.bernoulli(torch.cuda.FloatTensor(10).fill_(0.2))
    # print(bla)
    # print(bla.sum() / bla.size(0))
    # func = ffi.()
    # # return getattr(ffi, 'cluster_{}{}'.format(name, cuda))
    # print('drin')
    # pass
