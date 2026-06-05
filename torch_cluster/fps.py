from typing import List, Optional, Union

import torch
from torch import Tensor

import torch_cluster.typing


@torch.library.register_fake("torch_cluster::fps")
def _(src, ptr, ratio, random_start=True):
    torch._check(src.device == ptr.device)
    torch._check(ptr.ndim == 1)

    ctx = torch.library.get_ctx()
    nnz = ctx.new_dynamic_size()
    return ptr.new_empty((nnz,))


def fps(  # noqa
    src: torch.Tensor,
    batch: Optional[Tensor] = None,
    ratio: Optional[Union[Tensor, float]] = None,
    random_start: bool = True,
    batch_size: Optional[int] = None,
    ptr: Optional[Union[Tensor, List[int]]] = None,
):
    r""""A sampling algorithm from the `"PointNet++: Deep Hierarchical Feature
    Learning on Point Sets in a Metric Space"
    <https://arxiv.org/abs/1706.02413>`_ paper, which iteratively samples the
    most distant point with regard to the rest points.

    Args:
        src (Tensor): Point feature matrix
            :math:`\mathbf{X} \in \mathbb{R}^{N \times F}`.
        batch (LongTensor, optional): Batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            node to a specific example. (default: :obj:`None`)
        ratio (float or Tensor, optional): Sampling ratio.
            (default: :obj:`0.5`)
        random_start (bool, optional): If set to :obj:`False`, use the first
            node in :math:`\mathbf{X}` as starting node. (default: obj:`True`)
        batch_size (int, optional): The number of examples :math:`B`.
            Automatically calculated if not given. (default: :obj:`None`)
        ptr (torch.Tensor or [int], optional): If given, batch assignment will
            be determined based on boundaries in CSR representation, *e.g.*,
            :obj:`batch=[0,0,1,1,1,2]` translates to :obj:`ptr=[0,2,5,6]`.
            (default: :obj:`None`)

    :rtype: :class:`LongTensor`

    .. code-block:: python

        import torch
        from torch_cluster import fps

        src = torch.Tensor([[-1, -1], [-1, 1], [1, -1], [1, 1]])
        batch = torch.tensor([0, 0, 0, 0])
        index = fps(src, batch, ratio=0.5)
    """
    r: Optional[Tensor] = None
    if ratio is None:
        r = torch.tensor(0.5, dtype=src.dtype, device=src.device)
    elif isinstance(ratio, float):
        r = torch.tensor(ratio, dtype=src.dtype, device=src.device)
    else:
        r = ratio
    assert r is not None

    if ptr is not None:
        if isinstance(ptr, list) and torch_cluster.typing.WITH_PTR_LIST:
            return torch.ops.torch_cluster.fps_ptr_list(
                src, ptr, r, random_start)

        if isinstance(ptr, list):
            return torch.ops.torch_cluster.fps(
                src, torch.tensor(ptr, device=src.device), r, random_start)
        else:
            return torch.ops.torch_cluster.fps(src, ptr, r, random_start)

    if batch is not None:
        assert src.size(0) == batch.numel()
        if batch_size is None:
            batch_size = int(batch.max()) + 1

        deg = src.new_zeros(batch_size, dtype=torch.long)
        deg.scatter_add_(0, batch, torch.ones_like(batch))

        ptr_vec = deg.new_zeros(batch_size + 1)
        torch.cumsum(deg, 0, out=ptr_vec[1:])
    else:
        ptr_vec = torch.tensor([0, src.size(0)], device=src.device)

    return torch.ops.torch_cluster.fps(src, ptr_vec, r, random_start)
