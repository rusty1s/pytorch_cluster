from .utils import get_func, consecutive
from .degree import node_degree
from .permute import permute


def random_cluster(edge_index,
                   batch=None,
                   rid=None,
                   perm_edges=True,
                   num_nodes=None):

    num_nodes = edge_index.max() + 1 if num_nodes is None else num_nodes
    row, col = permute(edge_index, num_nodes, rid, perm_edges)
    degree = node_degree(row, num_nodes, out=row.new())

    cluster = edge_index.new(num_nodes).fill_(-1)
    func = get_func('random', cluster)
    func(cluster, row, col, degree)

    cluster, u = consecutive(cluster)

    if batch is None:
        return cluster
    else:
        # TODO: Fix
        return cluster, batch
