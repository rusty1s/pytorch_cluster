import torch


def randperm(row, col):
    # Randomly reorder row and column indices.
    edge_rid = torch.randperm(row.size(0))
    return row[edge_rid], col[edge_rid]


def sort_row(row, col):
    # Sort row and column indices row-wise.
    row, perm = row.sort()
    col = col[perm]
    return row, col


def randperm_sort_row(row, col, num_nodes):
    # Randomly change row indices to new values.
    node_rid = torch.randperm(num_nodes)
    row = node_rid[row]

    # Sort row and column indices row-wise.
    row, col = sort_row(row, col)

    # Revert previous row value changes to old indices.
    row = node_rid.sort()[1][row]

    return row, col
