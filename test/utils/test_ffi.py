import torch
from torch_cluster.functions.utils.ffi import ffi_serial


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
