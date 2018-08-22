from itertools import product

import pytest
from torch_cluster import grid_cluster

from .utils import dtypes, devices, tensor

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


@pytest.mark.parametrize('test,dtype,device', product(tests, dtypes, devices))
def test_grid_cluster(test, dtype, device):
    pos = tensor(test['pos'], dtype, device)
    size = tensor(test['size'], dtype, device)
    start = tensor(test.get('start'), dtype, device)
    end = tensor(test.get('end'), dtype, device)

    cluster = grid_cluster(pos, size, start, end)
    assert cluster.tolist() == test['cluster']
