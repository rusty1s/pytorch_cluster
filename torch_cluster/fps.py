import torch

if torch.cuda.is_available():
    import fps_cuda


def fps(x, batch=None, ratio=0.5, random_start=True):
    """A clustering algorithm, which overlays a regular grid of user-defined
    size over a point cloud and clusters all points within a voxel.

    Args:
        x (Tensor): D-dimensional node features.
        batch (LongTensor, optional): Vector that maps each node to a graph.
            If :obj:`None`, all node features belong to the same graph. If not
            :obj:`None`, nodes of the same graph need to have contiguous memory
            layout and :obj:`batch` needs to be ascending.
            (default: :obj:`None`)
        ratio (float, optional): Sampling ratio. (default: :obj:`0.5`)
        random_start (bool, optional): Whether the starting node is
            sampled randomly. (default: :obj:`True`)

    Examples::

        >>> x = torch.Tensor([[-1, -1], [-1, 1], [1, -1], [1, 1]])
        >>> batch = torch.Tensor([0, 0, 0, 0])
        >>> sample = fps(pos, batch)
    """

    assert x.is_cuda
    assert x.dim() <= 2 and batch.dim() == 1
    assert x.size(0) == batch.size(0)

    x = x.view(-1, 1) if x.dim() == 1 else x

    if batch is None:
        batch = x.new_zeros(x.size(0), dtype=torch.long)

    op = fps_cuda.fps if x.is_cuda else None
    out = op(x, batch, ratio, random_start)

    return out
