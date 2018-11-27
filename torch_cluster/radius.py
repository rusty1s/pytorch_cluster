import torch

if torch.cuda.is_available():
    import radius_cuda


def radius(x, y, r, batch_x=None, batch_y=None, max_num_neighbors=32):
    """Finds for each element in `y` all points in `x` within distance `r`.

    Args:
        x (Tensor): D-dimensional point features.
        y (Tensor): D-dimensional point features.
        r (float): The radius.
        batch_x (LongTensor, optional): Vector that maps each point to its
            example identifier. If :obj:`None`, all points belong to the same
            example. If not :obj:`None`, points in the same example need to
            have contiguous memory layout and :obj:`batch` needs to be
            ascending. (default: :obj:`None`)
        batch_y (LongTensor, optional): See `batch_x` (default: :obj:`None`)
        max_num_neighbors (int, optional): The maximum number of neighbors to
            return for each element in `y`. (default: :obj:`32`)

    :rtype: :class:`LongTensor`

    Examples::

        >>> x = torch.Tensor([[-1, -1], [-1, 1], [1, -1], [1, 1]])
        >>> batch_x = torch.Tensor([0, 0, 0, 0])
        >>> y = torch.Tensor([[-1, 0], [1, 0]])
        >>> batch_x = torch.Tensor([0, 0])
        >>> out = radius(x, y, 1.5, batch_x, batch_y)
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

    op = radius_cuda.radius if x.is_cuda else None
    assign_index = op(x, y, r, batch_x, batch_y, max_num_neighbors)

    return assign_index


def radius_graph(x, r, batch=None, loop=False, max_num_neighbors=32):
    """Finds for each element in `x` all points in `x` within distance `r`.

    Args:
        x (Tensor): D-dimensional point features.
        r (float): The radius.
        batch (LongTensor, optional): Vector that maps each point to its
            example identifier. If :obj:`None`, all points belong to the same
            example. If not :obj:`None`, points in the same example need to
            have contiguous memory layout and :obj:`batch` needs to be
            ascending. (default: :obj:`None`)
        loop (bool, optional): If :obj:`True`, the graph will contain
            self-loops. (default: :obj:`False`)
        max_num_neighbors (int, optional): The maximum number of neighbors to
            return for each element in `y`. (default: :obj:`32`)

    :rtype: :class:`LongTensor`

    Examples::

        >>> x = torch.Tensor([[-1, -1], [-1, 1], [1, -1], [1, 1]])
        >>> batch = torch.Tensor([0, 0, 0, 0])
        >>> out = radius_graph(x, 1.5, batch)
    """

    edge_index = radius(x, x, r, batch, batch, max_num_neighbors + 1)
    row, col = edge_index
    if not loop:
        row, col = edge_index
        mask = row != col
        row, col = row[mask], col[mask]
        edge_index = torch.stack([row, col], dim=0)
    return edge_index
