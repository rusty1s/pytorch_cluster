import torch
from torch_cluster.functions.utils.consecutive import consecutive


def test_consecutive():
    vec = torch.LongTensor([0, 2, 3])
    assert consecutive(vec).tolist() == [0, 1, 2]

    vec = torch.LongTensor([0, 3, 2, 2, 3])
    assert consecutive(vec).tolist() == [0, 2, 1, 1, 2]

    vec = torch.LongTensor([0, 3, 2, 2, 3])
    assert consecutive(vec, return_unique=True)[0].tolist() == [0, 2, 1, 1, 2]
    assert consecutive(vec, return_unique=True)[1].tolist() == [0, 2, 3]
