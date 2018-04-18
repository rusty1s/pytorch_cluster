def remove_self_loops(row, col):
    mask = row != col
    return row[mask], col[mask]
