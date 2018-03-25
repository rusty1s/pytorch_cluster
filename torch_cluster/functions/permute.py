import torch


def permute(edge_index, num_nodes, node_rid=None, edge_rid=None):
    row, col = edge_index

    edge_rid = torch.randperm(row.size(0)) if edge_rid is None else edge_rid
    row, col = row[edge_rid], col[edge_rid]

    node_rid = torch.randperm(num_nodes) if node_rid is None else node_rid
    _, perm = node_rid[row].sort()
    row, col = row[perm], col[perm]

    return row, col
