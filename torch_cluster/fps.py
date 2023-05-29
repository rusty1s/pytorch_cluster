from typing import Optional, Union, List

import torch
from torch import Tensor


@torch.jit._overload  # noqa
def fps(src, batch, ratio, random_start, batch_size, ptr):  # noqa
    # type: (Tensor, Optional[Tensor], Optional[float], bool, Optional[int], Optional[Tensor]) -> Tensor  # noqa
    pass  # pragma: no cover


@torch.jit._overload  # noqa
def fps(src, batch, ratio, random_start, batch_size, ptr):  # noqa
    # type: (Tensor, Optional[Tensor], Optional[Tensor], bool, Optional[int], Optional[Tensor]) -> Tensor  # noqa
    pass  # pragma: no cover


@torch.jit._overload  # noqa
def fps(src, batch, ratio, random_start, batch_size, ptr):  # noqa
    # type: (Tensor, Optional[Tensor], Optional[float], bool, Optional[int], Optional[List[int]]) -> Tensor  # noqa
    pass  # pragma: no cover


@torch.jit._overload  # noqa
def fps(src, batch, ratio, random_start, batch_size, ptr):  # noqa
    # type: (Tensor, Optional[Tensor], Optional[Tensor], bool, Optional[int], Optional[List[int]]) -> Tensor  # noqa
    pass  # pragma: no cover


@torch.jit._overload  # noqa
def fps(src, batch, ratio, random_start, batch_size, ptr):  # noqa
    # type: (Tensor, Optional[Tensor], Optional[float], bool, Optional[int], Optional[List[int]]) -> Tensor  # noqa
    pass  # pragma: no cover


def fps_ptr_tensor(
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
        ptr (LongTensor or list of ints): Ptr vector, which defines nodes
            ranges for consecutive batches, e.g. batch=[0,0,1,1,1,2] translates
            to ptr=[0,2,5,6].

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
    assert batch is None or ptr is None

    if batch is not None:
        assert src.size(0) == batch.numel()
        if batch_size is None:
            batch_size = int(batch.max()) + 1

        deg = src.new_zeros(batch_size, dtype=torch.long)
        deg.scatter_add_(0, batch, torch.ones_like(batch))

        p = deg.new_zeros(batch_size + 1)
        torch.cumsum(deg, 0, out=p[1:])
    else:
        if ptr is None:
            p = torch.tensor([0, src.size(0)], device=src.device)
        else:
            if isinstance(ptr, Tensor):
                p = ptr
            else:
                p = torch.tensor(ptr, device=src.device)

    return torch.ops.torch_cluster.fps(src, p, r, random_start)


def fps_ptr_list(
    src: torch.Tensor,
    batch: Optional[Tensor] = None,
    ratio: Optional[float] = None,
    random_start: bool = True,
    batch_size: Optional[int] = None,
    ptr: Optional[List[int]] = None,
):
    if ptr is not None:
        return torch.ops.torch_cluster.fps_ptr_list(src, ptr,
                                                    ratio, random_start)
    return fps_ptr_tensor(src, batch, ratio, random_start, batch_size, ptr)


fps = fps_ptr_list if hasattr(  # noqa
    torch.ops.torch_cluster, 'fp_ptr_list') else fps_ptr_tensor
