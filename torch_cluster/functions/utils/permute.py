import torch


def sort(edge_index):
    row, col = edge_index
    row, perm = row.sort()
    col = col[perm]
    return torch.stack([row, col], dim=0)


def permute(edge_index, num_nodes, node_rid=None, edge_rid=None):
    num_edges = edge_index.size(1)

    # Randomly reorder row and column indices.
    if edge_rid is None:
        edge_rid = torch.randperm(num_edges).type_as(edge_index)
    row, col = edge_index[:, edge_rid]

    # Randomly change row indices to new values.
    if node_rid is None:
        node_rid = torch.randperm(num_nodes).type_as(edge_index)
    row = node_rid[row]

    # Sort row and column indices based on changed values.
    row, col = sort(torch.stack([row, col], dim=0))

    # Revert previous row value changes to old indices.
    row = node_rid.sort()[1][row]

    return torch.stack([row, col], dim=0)
