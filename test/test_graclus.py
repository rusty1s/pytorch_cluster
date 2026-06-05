from itertools import product

import pytest
import torch
from torch_cluster import graclus_cluster
from torch_cluster.testing import devices, dtypes, tensor

tests = [{
    'row': [0, 0, 1, 1, 1, 2, 2, 2, 3, 3],
    'col': [1, 2, 0, 2, 3, 0, 1, 3, 1, 2],
}, {
    'row': [0, 0, 1, 1, 1, 2, 2, 2, 3, 3],
    'col': [1, 2, 0, 2, 3, 0, 1, 3, 1, 2],
    'weight': [1, 2, 1, 3, 2, 2, 3, 1, 2, 1],
}]


def assert_correct(row, col, cluster):
    row, col, cluster = row.to('cpu'), col.to('cpu'), cluster.to('cpu')
    n = cluster.size(0)

    # Every node was assigned a cluster.
    assert cluster.min() >= 0

    # There are no more than two nodes in each cluster.
    _, index = torch.unique(cluster, return_inverse=True)
    count = torch.zeros_like(cluster)
    count.scatter_add_(0, index, torch.ones_like(cluster))
    assert (count > 2).max() == 0

    # Cluster value is minimal.
    assert (cluster <= torch.arange(n, dtype=cluster.dtype)).sum() == n

    # Corresponding clusters must be adjacent.
    for i in range(n):
        x = cluster[col[row == i]] == cluster[i]  # Neighbors with same cluster
        y = cluster == cluster[i]  # Nodes with same cluster.
        y[i] = 0  # Do not look at cluster of `i`.
        assert x.sum() == y.sum()


@pytest.mark.parametrize('test,dtype,device', product(tests, dtypes, devices))
def test_graclus_cluster(test, dtype, device):
    if dtype == torch.bfloat16 and device == torch.device('cuda:0'):
        return

    row = tensor(test['row'], torch.long, device)
    col = tensor(test['col'], torch.long, device)
    weight = tensor(test.get('weight'), dtype, device)

    cluster = graclus_cluster(row, col, weight)
    assert_correct(row, col, cluster)

    # graclus is not torch.compile compatible.
