import pytest
import torch
from torch_cluster.functions.utils.permute import sort, permute


def test_sort_cpu():
    edge_index = torch.LongTensor([
        [0, 1, 0, 2, 1, 2, 1, 3, 2, 3],
        [1, 0, 2, 0, 2, 1, 3, 1, 3, 2],
    ])
    expected_edge_index = [
        [0, 0, 1, 1, 1, 2, 2, 2, 3, 3],
        [1, 2, 0, 2, 3, 0, 1, 3, 1, 2],
    ]
    assert sort(edge_index).tolist() == expected_edge_index


def test_permute_cpu():
    edge_index = torch.LongTensor([
        [0, 1, 0, 2, 1, 2, 1, 3, 2, 3],
        [1, 0, 2, 0, 2, 1, 3, 1, 3, 2],
    ])
    node_rid = torch.LongTensor([2, 1, 3, 0])
    edge_rid = torch.LongTensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    edge_index = permute(edge_index, 4, node_rid, edge_rid)
    expected_edge_index = [
        [3, 3, 1, 1, 1, 0, 0, 2, 2, 2],
        [1, 2, 0, 2, 3, 1, 2, 0, 1, 3],
    ]

    assert edge_index.tolist() == expected_edge_index


@pytest.mark.skipif(not torch.cuda.is_available(), reason='no CUDA')
def test_sort_gpu():  # pragma: no cover
    edge_index = torch.cuda.LongTensor([
        [0, 1, 0, 2, 1, 2, 1, 3, 2, 3],
        [1, 0, 2, 0, 2, 1, 3, 1, 3, 2],
    ])
    expected_row = [0, 0, 1, 1, 1, 2, 2, 2, 3, 3]
    assert sort(edge_index)[0].cpu().tolist() == expected_row


@pytest.mark.skipif(not torch.cuda.is_available(), reason='no CUDA')
def test_permute_gpu():  # pragma: no cover
    edge_index = torch.cuda.LongTensor([
        [0, 1, 0, 2, 1, 2, 1, 3, 2, 3],
        [1, 0, 2, 0, 2, 1, 3, 1, 3, 2],
    ])
    node_rid = torch.cuda.LongTensor([2, 1, 3, 0])

    edge_index = permute(edge_index, 4, node_rid)
    expected_row = [3, 3, 1, 1, 1, 0, 0, 2, 2, 2]

    assert edge_index[0].cpu().tolist() == expected_row
