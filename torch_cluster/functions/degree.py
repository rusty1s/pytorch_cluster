import torch


def node_degree(edge_index, num_nodes, out=None):
    zero = torch.zeros(num_nodes, out=out)
    one = torch.ones(edge_index.size(1), out=zero.new())
    return zero.scatter_add_(0, edge_index[0], one)
