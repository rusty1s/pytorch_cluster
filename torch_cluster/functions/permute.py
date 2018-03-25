import torch


def random_permute(edge_index, num_nodes):
    row, col = edge_index

    rid = torch.randperm(row.size(0))
    row, col = row[rid], col[rid]

    _, perm = rid[torch.randperm(num_nodes)].sort()
    row, col = row[perm], col[perm]

    return torch.stack([row, col], dim=0)
