import torch

if torch.cuda.is_available():
    import knn_cuda


def knn(x, y, k, batch_x=None, batch_y=None):
    """Finds for each element in `y` the `k` nearest points in `x`.

    Args:
        x (Tensor): D-dimensional point features.
        y (Tensor): D-dimensional point features.
        k (int): The number of neighbors.
        batch_x (LongTensor, optional): Vector that maps each point to its
            example identifier. If :obj:`None`, all points belong to the same
            example. If not :obj:`None`, points in the same example need to
            have contiguous memory layout and :obj:`batch` needs to be
            ascending. (default: :obj:`None`)
        batch_y (LongTensor, optional): See `batch_x` (default: :obj:`None`)

    :rtype: :class:`LongTensor`

    Examples::

        >>> x = torch.Tensor([[-1, -1], [-1, 1], [1, -1], [1, 1]])
        >>> batch_x = torch.Tensor([0, 0, 0, 0])
        >>> y = torch.Tensor([[-1, 0], [1, 0]])
        >>> batch_x = torch.Tensor([0, 0])
        >>> out = knn(x, y, 2, batch_x, batch_y)
    """

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

    op = knn_cuda.knn if x.is_cuda else None
    assign_index = op(x, y, k, batch_x, batch_y)

    return assign_index


def knn_graph(x, k, batch=None, loop=False):
    """Finds for each element in `x` the `k` nearest points.

    Args:
        x (Tensor): D-dimensional point features.
        k (int): The number of neighbors.
        batch (LongTensor, optional): Vector that maps each point to its
            example identifier. If :obj:`None`, all points belong to the same
            example. If not :obj:`None`, points in the same example need to
            have contiguous memory layout and :obj:`batch` needs to be
            ascending. (default: :obj:`None`)
        loop (bool, optional): If :obj:`True`, the graph will contain
            self-loops. (default: :obj:`False`)

    :rtype: :class:`LongTensor`

    Examples::

        >>> x = torch.Tensor([[-1, -1], [-1, 1], [1, -1], [1, 1]])
        >>> batch = torch.Tensor([0, 0, 0, 0])
        >>> out = knn_graph(x, 2, batch)
    """

    edge_index = knn(x, x, k if loop else k + 1, batch, batch)
    if not loop:
        row, col = edge_index
        mask = row != col
        row, col = row[mask], col[mask]
        edge_index = torch.stack([row, col], dim=0)
    return edge_index
