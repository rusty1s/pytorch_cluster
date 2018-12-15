import torch
import scipy.spatial

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
        >>> batch_x = torch.tensor([0, 0, 0, 0])
        >>> y = torch.Tensor([[-1, 0], [1, 0]])
        >>> batch_x = torch.tensor([0, 0])
        >>> assign_index = radius(x, y, 1.5, batch_x, batch_y)
    """

    if batch_x is None:
        batch_x = x.new_zeros(x.size(0), dtype=torch.long)

    if batch_y is None:
        batch_y = y.new_zeros(y.size(0), dtype=torch.long)

    x = x.view(-1, 1) if x.dim() == 1 else x
    y = y.view(-1, 1) if y.dim() == 1 else y

    assert x.dim() == 2 and batch_x.dim() == 1
    assert y.dim() == 2 and batch_y.dim() == 1
    assert x.size(1) == y.size(1)
    assert x.size(0) == batch_x.size(0)
    assert y.size(0) == batch_y.size(0)

    if x.is_cuda:
        return radius_cuda.radius(x, y, r, batch_x, batch_y, max_num_neighbors)

    x = torch.cat([x, 2 * r * batch_x.view(-1, 1).to(x.dtype)], dim=-1)
    y = torch.cat([y, 2 * r * batch_y.view(-1, 1).to(y.dtype)], dim=-1)

    tree = scipy.spatial.cKDTree(x)
    col = tree.query_ball_point(y, r)
    col = [torch.tensor(c) for c in col]
    row = [torch.full_like(c, i) for i, c in enumerate(col)]
    row, col = torch.cat(row, dim=0), torch.cat(col, dim=0)

    return torch.stack([row, col], dim=0)


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
        >>> batch = torch.tensor([0, 0, 0, 0])
        >>> edge_index = radius_graph(x, r=1.5, batch=batch, loop=False)
    """

    edge_index = radius(x, x, r, batch, batch, max_num_neighbors + 1)
    row, col = edge_index
    if not loop:
        row, col = edge_index
        mask = row != col
        row, col = row[mask], col[mask]
        edge_index = torch.stack([row, col], dim=0)
    return edge_index
