from itertools import product

import pytest
import torch
from torch_cluster import grid_cluster

from .tensor import tensors

tests = [{
    'pos': [2, 6],
    'size': [5],
    'cluster': [0, 0],
}, {
    'pos': [2, 6],
    'size': [5],
    'start': [0],
    'cluster': [0, 1],
}, {
    'pos': [[0, 0], [11, 9], [2, 8], [2, 2], [8, 3]],
    'size': [5, 5],
    'cluster': [0, 5, 3, 0, 1],
}, {
    'pos': [[0, 0], [11, 9], [2, 8], [2, 2], [8, 3]],
    'size': [5, 5],
    'end': [19, 19],
    'cluster': [0, 6, 4, 0, 1],
}]


@pytest.mark.parametrize('tensor,i', product(tensors, range(len(tests))))
def test_grid_cluster_cpu(tensor, i):
    data = tests[i]

    pos = getattr(torch, tensor)(data['pos'])
    size = getattr(torch, tensor)(data['size'])

    start = data.get('start')
    start = start if start is None else getattr(torch, tensor)(start)

    end = data.get('end')
    end = end if end is None else getattr(torch, tensor)(end)

    cluster = grid_cluster(pos, size, start, end)
    assert cluster.tolist() == data['cluster']


@pytest.mark.skipif(not torch.cuda.is_available(), reason='no CUDA')
@pytest.mark.parametrize('tensor,i', product(tensors, range(len(tests))))
def test_grid_cluster_gpu(tensor, i):  # pragma: no cover
    data = tests[i]

    pos = getattr(torch.cuda, tensor)(data['pos'])
    size = getattr(torch.cuda, tensor)(data['size'])

    start = data.get('start')
    start = start if start is None else getattr(torch.cuda, tensor)(start)

    end = data.get('end')
    end = end if end is None else getattr(torch.cuda, tensor)(end)

    cluster = grid_cluster(pos, size, start, end)
    assert cluster.tolist() == data['cluster']
