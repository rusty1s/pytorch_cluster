import torch
from torch_cluster.functions.utils.degree import node_degree


def test_node_degree():
    row = torch.LongTensor([0, 1, 1, 0, 0, 3, 0])
    expected_degree = [4, 2, 0, 1]

    degree = node_degree(row, 4)
    assert degree.type() == torch.FloatTensor().type()
    assert degree.tolist() == expected_degree

    degree = node_degree(row, 4, out=torch.LongTensor())
    assert degree.type() == torch.LongTensor().type()
    assert degree.tolist() == expected_degree
