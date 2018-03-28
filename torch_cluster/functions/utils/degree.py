import torch


def node_degree(target, num_nodes, out=None):
    out = target.new(num_nodes) if out is None else out
    zero = torch.zeros(num_nodes, out=out)
    one = torch.ones(target.size(0), out=zero.new(target.size(0)))
    return zero.scatter_add_(0, target, one)
