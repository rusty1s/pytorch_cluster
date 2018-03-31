import torch
import numpy as np
from torch_cluster import graclus_cluster


def assert_correct_graclus(row, col, cluster):
    row, col, cluster = row.numpy(), col.numpy(), cluster.numpy()

    # Every node was assigned a cluster.
    assert cluster.min() >= 0

    # There are no more than two nodes in each cluster.
    _, count = np.unique(cluster, return_counts=True)
    assert (count > 2).max() == 0

    # Corresponding clusters must be adjacent.
    for n in range(cluster.shape[0]):
        assert (cluster[col[row == n]] == cluster[n]).max() == 1


def test_graclus_cluster_cpu():
    row = torch.LongTensor([0, 0, 1, 1, 1, 2, 2, 2, 3, 3])
    col = torch.LongTensor([1, 2, 0, 2, 3, 0, 1, 3, 1, 2])
    weight = torch.Tensor([1, 2, 1, 3, 2, 2, 3, 1, 2, 1])

    cluster = graclus_cluster(row, col)
    assert_correct_graclus(row, col, cluster)

    cluster = graclus_cluster(row, col, weight)
    assert_correct_graclus(row, col, cluster)
