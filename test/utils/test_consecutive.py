import pytest
import torch
from torch_cluster.functions.utils.consecutive import consecutive


def test_consecutive_cpu():
    x = torch.LongTensor([[0, 3, 2], [2, 3, 0]])
    assert consecutive(x).tolist() == [[0, 2, 1], [1, 2, 0]]


@pytest.mark.skipif(not torch.cuda.is_available(), reason='no CUDA')
def test_consecutive_gpu():  # pragma: no cover
    x = torch.cuda.LongTensor([[0, 3, 2], [2, 3, 0]])
    assert consecutive(x).cpu().tolist() == [[0, 2, 1], [1, 2, 0]]
