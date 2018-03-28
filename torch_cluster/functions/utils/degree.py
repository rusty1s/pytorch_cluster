import torch


def node_degree(row, num_nodes, out=None):
    out = row.new() if out is None else out
    zero = torch.zeros(num_nodes, out=out)
    one = torch.ones(row.size(0), out=zero.new())
    return zero.scatter_add_(0, row, one)
