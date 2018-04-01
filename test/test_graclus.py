from itertools import product

import pytest
import torch
import numpy as np
from torch_cluster import graclus_cluster

from .tensor import cpu_tensors

tests = [{
    'row': [0, 0, 1, 1, 1, 2, 2, 2, 3, 3],
    'col': [1, 2, 0, 2, 3, 0, 1, 3, 1, 2],
    'weight': None,
}, {
    'row': [0, 0, 1, 1, 1, 2, 2, 2, 3, 3],
    'col': [1, 2, 0, 2, 3, 0, 1, 3, 1, 2],
    'weight': [1, 2, 1, 3, 2, 2, 3, 1, 2, 1],
}]


def assert_correct_graclus(row, col, cluster):
    row, col, cluster = row.numpy(), col.numpy(), cluster.numpy()

    # Every node was assigned a cluster.
    assert cluster.min() >= 0

    # There are no more than two nodes in each cluster.
    _, count = np.unique(cluster, return_counts=True)
    assert (count > 2).max() == 0

    # Corresponding clusters must be adjacent.
    for n in range(cluster.shape[0]):
        x = cluster[col[row == n]] == cluster[n]  # Neighbors with same cluster
        y = cluster == cluster[n]  # Nodes with same cluster
        y[n] = 0  # Do not look at cluster of node `n`.
        assert x.sum() == y.sum()


@pytest.mark.parametrize('tensor,i', product(cpu_tensors, range(len(tests))))
def test_graclus_cluster_cpu(tensor, i):
    data = tests[i]

    row = torch.LongTensor(data['row'])
    col = torch.LongTensor(data['col'])

    weight = data['weight']
    weight = weight if weight is None else getattr(torch, tensor)(weight)

    cluster = graclus_cluster(row, col, weight)
    assert_correct_graclus(row, col, cluster)
