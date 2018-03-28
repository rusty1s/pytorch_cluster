import pytest
import torch
from torch_cluster.functions.utils.consecutive import consecutive


def test_consecutive():
    vec = torch.LongTensor([0, 2, 3])
    assert consecutive(vec).tolist() == [0, 1, 2]

    vec = torch.LongTensor([0, 3, 2, 2, 3])
    assert consecutive(vec).tolist() == [0, 2, 1, 1, 2]

    vec = torch.LongTensor([0, 3, 2, 2, 3])
    assert consecutive(vec, True)[0].tolist() == [0, 2, 1, 1, 2]
    assert consecutive(vec, True)[1].tolist() == [0, 2, 3]


@pytest.mark.skipif(not torch.cuda.is_available(), reason='no CUDA')
def test_consecutive_gpu():  # pragma: no cover
    vec = torch.cuda.LongTensor([0, 2, 3])
    assert consecutive(vec).cpu().tolist() == [0, 1, 2]

    vec = torch.cuda.LongTensor([0, 3, 2, 2, 3])
    assert consecutive(vec).cpu().tolist() == [0, 2, 1, 1, 2]

    vec = torch.cuda.LongTensor([0, 3, 2, 2, 3])
    assert consecutive(vec, True)[0].cpu().tolist() == [0, 2, 1, 1, 2]
    assert consecutive(vec, True)[1].cpu().tolist() == [0, 2, 3]
