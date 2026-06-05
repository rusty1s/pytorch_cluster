from itertools import product

import pytest
import torch
from torch_cluster import grid_cluster
from torch_cluster.testing import devices, dtypes, has_compiler, tensor

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
    if dtype == torch.bfloat16 and device == torch.device('cuda:0'):
        return

    pos = tensor(test['pos'], dtype, device)
    size = tensor(test['size'], dtype, device)
    start = tensor(test.get('start'), dtype, device)
    end = tensor(test.get('end'), dtype, device)

    cluster = grid_cluster(pos, size, start, end)
    assert cluster.tolist() == test['cluster']

    if has_compiler():
        jit = torch.compile(grid_cluster)
        assert torch.equal(jit(pos, size, start, end), cluster)
