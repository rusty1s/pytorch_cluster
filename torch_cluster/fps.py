from typing import List, Optional, Union

import torch
from torch import Tensor

import torch_cluster.typing


@torch.jit._overload  # noqa
def fps(src, batch, ratio, num_points, random_start, batch_size, ptr):  # noqa
    # type: (Tensor, Optional[Tensor], Optional[float], Optional[int], bool, Optional[int], Optional[Tensor]) -> Tensor  # noqa
    pass  # pragma: no cover


@torch.jit._overload  # noqa
def fps(src, batch, ratio, num_points, random_start, batch_size, ptr):  # noqa
    # type: (Tensor, Optional[Tensor], Optional[Tensor], Optional[int], bool, Optional[int], Optional[Tensor]) -> Tensor  # noqa
    pass  # pragma: no cover


@torch.jit._overload  # noqa
def fps(src, batch, ratio, num_points, random_start, batch_size, ptr):  # noqa
    # type: (Tensor, Optional[Tensor], Optional[float], Optional[int], bool, Optional[int], Optional[List[int]]) -> Tensor  # noqa
    pass  # pragma: no cover


@torch.jit._overload  # noqa
def fps(src, batch, ratio, num_points, random_start, batch_size, ptr):  # noqa
    # type: (Tensor, Optional[Tensor], Optional[Tensor], Optional[int], bool, Optional[int], Optional[List[int]]) -> Tensor  # noqa
    pass  # pragma: no cover

@torch.jit._overload  # noqa
def fps(src, batch, ratio, num_points, random_start, batch_size, ptr):  # noqa
    # type: (Tensor, Optional[Tensor], Optional[float], Optional[Tensor], bool, Optional[int], Optional[Tensor]) -> Tensor  # noqa
    pass  # pragma: no cover


@torch.jit._overload  # noqa
def fps(src, batch, ratio, num_points, random_start, batch_size, ptr):  # noqa
    # type: (Tensor, Optional[Tensor], Optional[Tensor], Optional[Tensor], bool, Optional[int], Optional[Tensor]) -> Tensor  # noqa
    pass  # pragma: no cover


@torch.jit._overload  # noqa
def fps(src, batch, ratio, num_points, random_start, batch_size, ptr):  # noqa
    # type: (Tensor, Optional[Tensor], Optional[float], Optional[Tensor], bool, Optional[int], Optional[List[int]]) -> Tensor  # noqa
    pass  # pragma: no cover


@torch.jit._overload  # noqa
def fps(src, batch, ratio, num_points, random_start, batch_size, ptr):  # noqa
    # type: (Tensor, Optional[Tensor], Optional[Tensor], Optional[Tensor], bool, Optional[int], Optional[List[int]]) -> Tensor  # noqa
    pass  # pragma: no cover


def fps(  # noqa
    src: torch.Tensor,
    batch: Optional[Tensor] = None,
    ratio: Optional[Union[Tensor, float]] = None,
    num_points: Optional[Union[Tensor, int]] = None,
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
            Only ratio or num_points can be specified.
            (default: :obj:`0.5`)
        num_points (int, optional): Number of returned points.
            Only ratio or num_points can be specified.
            (default: :obj:`None`)
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
    # check if only of of ratio or num_points is set
    # if no one is set, fallback to ratio = 0.5
    if ratio is not None and num_points is not None:
        raise ValueError("Only one of ratio and num_points can be specified.")

    r: Optional[Tensor] = None
    if ratio is None:
        if num_points is None:
            r = torch.tensor(0.5, dtype=src.dtype, device=src.device)
        else:
            r = torch.tensor(0.0, dtype=src.dtype, device=src.device)
    elif isinstance(ratio, float):
        r = torch.tensor(ratio, dtype=src.dtype, device=src.device)
    else:
        r = ratio

    num: Optional[Tensor] = None
    if num_points is None:
        num = torch.tensor(0, dtype=torch.long, device=src.device)
    elif isinstance(num_points, int):
        num = torch.tensor(num_points, dtype=torch.long, device=src.device)
    else:
        num = num_points

    assert r is not None and num is not None

    if r.sum() == 0 and num.sum() == 0:
        raise ValueError("At least one of ratio or num_points should be > 0")

    if ptr is not None:
        if isinstance(ptr, list) and torch_cluster.typing.WITH_PTR_LIST:
            return torch.ops.torch_cluster.fps_ptr_list(
                src, ptr, r, random_start
            )

        if isinstance(ptr, list):
            return torch.ops.torch_cluster.fps(
                src, torch.tensor(ptr, device=src.device), r, num, random_start
            )
        else:
            return torch.ops.torch_cluster.fps(src, ptr, r, num, random_start)

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

    return torch.ops.torch_cluster.fps(src, ptr_vec, r, num, random_start)
