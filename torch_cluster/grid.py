import torch
import grid_cpu

if torch.cuda.is_available():
    import grid_cuda


def grid_cluster(pos, size, start=None, end=None):
    """A clustering algorithm, which overlays a regular grid of user-defined
    size over a point cloud and clusters all points within a voxel.

    Args:
        pos (Tensor): D-dimensional position of points.
        size (Tensor): Size of a voxel in each dimension.
        start (Tensor, optional): Start position of the grid (in each
            dimension). (default: :obj:`None`)
        end (Tensor, optional): End position of the grid (in each
            dimension). (default: :obj:`None`)

    :rtype: :class:`LongTensor`

    Examples::

        >>> pos = torch.Tensor([[0, 0], [11, 9], [2, 8], [2, 2], [8, 3]])
        >>> size = torch.Tensor([5, 5])
        >>> cluster = grid_cluster(pos, size)
    """

    pos = pos.unsqueeze(-1) if pos.dim() == 1 else pos
    start = pos.t().min(dim=1)[0] if start is None else start
    end = pos.t().max(dim=1)[0] if end is None else end

    op = grid_cuda if pos.is_cuda else grid_cpu
    cluster = op.grid(pos, size, start, end)

    return cluster
