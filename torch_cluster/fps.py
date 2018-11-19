import torch

if torch.cuda.is_available():
    import fps_cuda
    """    """


def fps(x, batch=None, ratio=0.5, random_start=True):
    """Samples a specified ratio of points for each element in a batch using
    farthest iterative point sampling.

    Args:
        x (Tensor): D-dimensional point features.
        batch (LongTensor, optional): Vector that maps each point to its
            example identifier. If :obj:`None`, all points belong to the same
            example. If not :obj:`None`, points in the same example need to
            have contiguous memory layout and :obj:`batch` needs to be
            ascending. (default: :obj:`None`)
        ratio (float, optional): Sampling ratio. (default: :obj:`0.5`)
        random_start (bool, optional): Whether the starting node is
            sampled randomly. (default: :obj:`True`)

    :rtype: :class:`LongTensor`

    Examples::

        >>> x = torch.Tensor([[-1, -1], [-1, 1], [1, -1], [1, 1]])
        >>> batch = torch.Tensor([0, 0, 0, 0])
        >>> sample = fps(x, batch)
    """

    if batch is None:
        batch = x.new_zeros(x.size(0), dtype=torch.long)

    x = x.view(-1, 1) if x.dim() == 1 else x

    assert x.is_cuda
    assert x.dim() == 2 and batch.dim() == 1
    assert x.size(0) == batch.size(0)
    assert ratio > 0 and ratio < 1

    op = fps_cuda.fps if x.is_cuda else None
    out = op(x, batch, ratio, random_start)

    return out
