from .utils.permute import permute
from .utils.degree import node_degree
from .utils.ffi import _get_func
from .utils.consecutive import consecutive


def serial_cluster(edge_index, batch=None, num_nodes=None):

    num_nodes = edge_index.max() + 1 if num_nodes is None else num_nodes
    row, col = permute(edge_index, num_nodes)
    degree = node_degree(row, num_nodes, out=row.new())

    cluster = edge_index.new(num_nodes).fill_(-1)
    func = _get_func('random', cluster)
    func(cluster, row, col, degree)

    cluster, u = consecutive(cluster)

    if batch is None:
        return cluster
    else:
        # TODO: Fix
        return cluster, batch
