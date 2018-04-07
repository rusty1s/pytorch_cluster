from .utils.perm import randperm, sort_row, randperm_sort_row
from .utils.ffi import graclus


def graclus_cluster(row, col, weight=None, num_nodes=None):
    num_nodes = row.max() + 1 if num_nodes is None else num_nodes

    if row.is_cuda:  # pragma: no cover
        row, col = sort_row(row, col)
    else:
        row, col = randperm(row, col)
        row, col = randperm_sort_row(row, col, num_nodes)

    cluster = row.new(num_nodes)
    graclus(cluster, row, col, weight)

    return cluster
