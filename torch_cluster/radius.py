import torch

if torch.cuda.is_available():
    import radius_cuda


def radius(x, y, r, batch_x=None, batch_y=None, max_num_neighbors=32):

    if batch_x is None:
        batch_x = x.new_zeros(x.size(0), dtype=torch.long)

    if batch_y is None:
        batch_y = y.new_zeros(y.size(0), dtype=torch.long)

    x = x.view(-1, 1) if x.dim() == 1 else x
    y = y.view(-1, 1) if y.dim() == 1 else y

    assert x.is_cuda
    assert x.dim() == 2 and batch_x.dim() == 1
    assert y.dim() == 2 and batch_y.dim() == 1
    assert x.size(1) == y.size(1)
    assert x.size(0) == batch_x.size(0)
    assert y.size(0) == batch_y.size(0)

    op = radius_cuda.radius if x.is_cuda else None
    assign_index = op(x, y, r, batch_x, batch_y, max_num_neighbors)

    return assign_index


def radius_graph(x, r, batch=None, max_num_neighbors=32):
    edge_index = radius(x, x, r, batch, batch, max_num_neighbors + 1)
    row, col = edge_index
    mask = row != col
    row, col = row[mask], col[mask]
    return torch.stack([row, col], dim=0)
