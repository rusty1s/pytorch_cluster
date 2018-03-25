from __future__ import division

import torch


def normalized_cut(edge_index, num_nodes, degree, edge_attr=None):
    row, col = edge_index

    cut = 1 / degree
    cut = cut[row] + cut[col]

    if edge_attr is None:
        return cut
    else:
        if edge_attr.dim() > 1 and edge_attr.size(1) > 1:
            edge_attr = torch.norm(edge_attr, 2, 1)
        return edge_attr.squeeze() * cut
