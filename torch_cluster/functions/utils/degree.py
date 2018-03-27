import torch


def node_degree(row, num_nodes, out=None):
    zero = torch.zeros(num_nodes, out=out)
    one = torch.ones(row.size(0), out=zero.new())
    return zero.scatter_add_(0, row, one)
