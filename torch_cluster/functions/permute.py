import torch


def permute(edge_index, num_nodes, rid=None, perm_edges=True):
    row, col = edge_index

    if perm_edges:
        edge_rid = torch.randperm(row.size(0))
        row, col = row[edge_rid], col[edge_rid]

    rid = torch.randperm(num_nodes) if rid is None else rid
    _, perm = rid[row].sort()
    row, col = row[perm], col[perm]

    return torch.stack([row, col], dim=0)
