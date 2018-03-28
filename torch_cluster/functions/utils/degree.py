import torch


def node_degree(index, num_nodes, out=None):
    out = index.new(num_nodes) if out is None else out
    zero = torch.zeros(num_nodes, out=out)
    one = torch.ones(index.size(0), out=zero.new())
    return zero.scatter_add_(0, index, one)
