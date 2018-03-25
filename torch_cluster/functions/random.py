from .utils import get_func
from .degree import node_degree
from .permute import permute


def random_cluster(edge_index, node_rid=None, edge_rid=None, num_nodes=None):
    num_nodes = edge_index.max() + 1 if num_nodes is None else num_nodes
    row, col = permute(edge_index, num_nodes, node_rid, edge_rid)
    degree = node_degree(row, num_nodes, out=row.new())

    cluster = edge_index.new(num_nodes).fill_(-1)
    func = get_func('random', cluster)
    func(cluster, row, col, degree)

    return cluster
