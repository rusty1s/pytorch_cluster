import pytest
import torch
from torch_cluster.functions.utils.permute import sort, permute


def test_sort_cpu():
    row = torch.LongTensor([0, 1, 0, 2, 1, 2, 1, 3, 2, 3])
    col = torch.LongTensor([1, 0, 2, 0, 2, 1, 3, 1, 3, 2])
    row, col = sort(row, col)
    expected_row = [0, 0, 1, 1, 1, 2, 2, 2, 3, 3]
    expected_col = [1, 2, 0, 2, 3, 0, 1, 3, 1, 2]
    assert row.tolist() == expected_row
    assert col.tolist() == expected_col


def test_permute_cpu():
    row = torch.LongTensor([0, 1, 0, 2, 1, 2, 1, 3, 2, 3])
    col = torch.LongTensor([1, 0, 2, 0, 2, 1, 3, 1, 3, 2])
    node_rid = torch.LongTensor([2, 1, 3, 0])
    edge_rid = torch.LongTensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    row, col = permute(row, col, 4, node_rid, edge_rid)
    expected_row = [3, 3, 1, 1, 1, 0, 0, 2, 2, 2]
    expected_col = [1, 2, 0, 2, 3, 1, 2, 0, 1, 3]
    assert row.tolist() == expected_row
    assert col.tolist() == expected_col


@pytest.mark.skipif(not torch.cuda.is_available(), reason='no CUDA')
def test_sort_gpu():  # pragma: no cover
    row = torch.cuda.LongTensor([0, 1, 0, 2, 1, 2, 1, 3, 2, 3])
    col = torch.cuda.LongTensor([1, 0, 2, 0, 2, 1, 3, 1, 3, 2])
    row, col = sort(row, col)
    expected_row = [0, 0, 1, 1, 1, 2, 2, 2, 3, 3]
    expected_col = [1, 2, 0, 2, 3, 0, 1, 3, 1, 2]
    assert row.cpu().tolist() == expected_row
    assert col.cpu().tolist() == expected_col


@pytest.mark.skipif(not torch.cuda.is_available(), reason='no CUDA')
def test_permute_gpu():  # pragma: no cover
    row = torch.cuda.LongTensor([0, 1, 0, 2, 1, 2, 1, 3, 2, 3])
    col = torch.cuda.LongTensor([1, 0, 2, 0, 2, 1, 3, 1, 3, 2])
    node_rid = torch.cuda.LongTensor([2, 1, 3, 0])
    edge_rid = torch.cuda.LongTensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    row, col = permute(row, col, 4, node_rid, edge_rid)
    expected_row = [3, 3, 1, 1, 1, 0, 0, 2, 2, 2]
    expected_col = [1, 2, 0, 2, 3, 1, 2, 0, 1, 3]
    assert row.cpu().tolist() == expected_row
    assert col.cpu().tolist() == expected_col
