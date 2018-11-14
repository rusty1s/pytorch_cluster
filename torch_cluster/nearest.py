import torch

if torch.cuda.is_available():
    import nearest_cuda


def nearest(x, y, batch_x=None, batch_y=None):
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

    op = nearest_cuda.nearest if x.is_cuda else None
    dist, idx = op(x, y, batch_x, batch_y)

    return dist, idx
