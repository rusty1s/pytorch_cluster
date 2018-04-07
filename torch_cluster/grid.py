from .utils.ffi import grid


def grid_cluster(pos, size, start=None, end=None):
    """A clustering algorithm, which overlays a regular grid of user-defined
    size over a point cloud and clusters all points within a voxel.

    Args:
        pos (Tensor): D-dimensional position of points.
        size (Tensor): Size of a voxel in each dimension.
        start (Tensor or int, optional): Start position of the grid (in each
            dimension). (default: :obj:`None`)
        end (Tensor or int, optional): End position of the grid (in each
            dimension). (default: :obj:`None`)

    Examples::

        >>> pos = torch.Tensor([[0, 0], [11, 9], [2, 8], [2, 2], [8, 3]])
        >>> size = torch.Tensor([5, 5])
        >>> cluster = grid_cluster(pos, size)
    """

    pos = pos.unsqueeze(-1) if pos.dim() == 1 else pos

    assert pos.size(1) == size.size(0), (
        'Last dimension of position tensor must have same size as size tensor')

    start = pos.t().min(dim=1)[0] if start is None else start
    end = pos.t().max(dim=1)[0] if end is None else end
    pos, end = pos - start, end - start

    size = size.type_as(pos)
    count = (end / size).long() + 1

    cluster = count.new(pos.size(0))
    grid(cluster, pos, size, count)

    return cluster
