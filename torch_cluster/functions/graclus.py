from __future__ import division

import torch

# from .utils import get_func
from .degree import node_degree


def graclus_cluster(edge_index, num_nodes=None, edge_attr=None, rid=None):
    num_nodes = edge_index.max() + 1 if num_nodes is None else num_nodes
    rid = torch.randperm(num_nodes) if rid is None else rid
    row, col = edge_index

    # Compute edge-wise normalized cut.
    cut = normalized_cut(edge_index, num_nodes, edge_attr)

    # Sort row and col indices based on the (possibly random) `rid`.
    _, perm = rid[row].sort()
    row, col, cut = row[perm], col[perm], cut[perm]
    print(row, col)

    cluster = edge_index.new(num_nodes).fill_(-1)
    # func = get_func('graclus', cluster)
    # func(cluster, row, col, cut)
    return cluster


def normalized_cut(edge_index, num_nodes, edge_attr=None):
    row, col = edge_index

    out = edge_attr.new() if edge_attr is not None else torch.Tensor()
    cut = node_degree(edge_index, num_nodes, out=out)
    cut = 1 / cut
    cut = cut[row] + cut[col]

    if edge_attr is None:
        return cut
    else:
        if edge_attr.dim() > 1 and edge_attr.size(1) > 1:
            edge_attr = torch.norm(edge_attr, 2, 1)
        return edge_attr.squeeze() * cut
