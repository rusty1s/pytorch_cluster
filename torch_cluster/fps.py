import torch
import fps_cpu

if torch.cuda.is_available():
    import fps_cuda


def fps(x, batch=None, ratio=0.5, random_start=True):
    r""""A sampling algorithm from the `"PointNet++: Deep Hierarchical Feature
    Learning on Point Sets in a Metric Space"
    <https://arxiv.org/abs/1706.02413>`_ paper, which iteratively samples the
    most distant point with regard to the rest points.

    Args:
        x (Tensor): Node feature matrix
            :math:`\mathbf{X} \in \mathbb{R}^{N \times F}`.
        batch (LongTensor, optional): Batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            node to a specific example. (default: :obj:`None`)
        ratio (float, optional): Sampling ratio. (default: :obj:`0.5`)
        random_start (bool, optional): If set to :obj:`False`, use the first
            node in :math:`\mathbf{X}` as starting node. (default: obj:`True`)

    :rtype: :class:`LongTensor`

    .. testsetup::

        import torch
        from torch_cluster import fps

    .. testcode::

        >>> x = torch.Tensor([[-1, -1], [-1, 1], [1, -1], [1, 1]])
        >>> batch = torch.tensor([0, 0, 0, 0])
        >>> index = fps(x, batch, ratio=0.5)
    """

    if batch is None:
        batch = x.new_zeros(x.size(0), dtype=torch.long)

    x = x.view(-1, 1) if x.dim() == 1 else x

    assert x.dim() == 2 and batch.dim() == 1
    assert x.size(0) == batch.size(0)
    assert ratio > 0 and ratio < 1

    if x.is_cuda:
        return fps_cuda.fps(x, batch, ratio, random_start)
    else:
        return fps_cpu.fps(x, batch, ratio, random_start)
