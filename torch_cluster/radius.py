import importlib.util
from typing import Optional

import torch


def _cap_neighbors_by_col(
    edge_index: torch.Tensor,
    max_num_neighbors: int,
) -> torch.Tensor:
    if max_num_neighbors <= 0 or edge_index.numel() == 0:
        return edge_index
    row, col = edge_index
    max_row = int(row.max().item()) if row.numel() > 0 else -1
    key = row * (max_row + 2) + col
    order = torch.argsort(key)
    row = row[order]
    col = col[order]
    idx = torch.arange(row.numel(), device=row.device)
    row_diff = torch.ones_like(row, dtype=torch.bool)
    if row.numel() > 1:
        row_diff[1:] = row[1:] != row[:-1]
    start_idx = torch.where(row_diff, idx, torch.zeros_like(idx))
    start_idx = torch.cummax(start_idx, 0).values
    pos = idx - start_idx
    mask = pos < max_num_neighbors
    row = row[mask]
    col = col[mask]
    return torch.stack([row, col], dim=0)


@torch.library.register_fake("torch_cluster::radius")
def _(
    x,
    y,
    ptr_x,
    ptr_y,
    r,
    max_num_neighbors=32,
    num_workers=1,
    ignore_same_index=False,
):
    torch._check(x.device == y.device)
    if ptr_x is not None:
        torch._check(ptr_x.device == x.device)
        torch._check(ptr_x.ndim == 1)
    if ptr_y is not None:
        torch._check(ptr_y.device == y.device)
        torch._check(ptr_y.ndim == 1)
    ctx = torch.library.get_ctx()
    nnz = ctx.new_dynamic_size()
    return x.new_empty((2, nnz), dtype=torch.long)


def radius(
    x: torch.Tensor,
    y: torch.Tensor,
    r: float,
    batch_x: Optional[torch.Tensor] = None,
    batch_y: Optional[torch.Tensor] = None,
    max_num_neighbors: int = 32,
    num_workers: int = 1,
    batch_size: Optional[int] = None,
    ignore_same_index: bool = False,
    use_triton: bool = False,
) -> torch.Tensor:
    r"""Finds for each element in :obj:`y` all points in :obj:`x` within
    distance :obj:`r`.

    Args:
        x (Tensor): Node feature matrix
            :math:`\mathbf{X} \in \mathbb{R}^{N \times F}`.
        y (Tensor): Node feature matrix
            :math:`\mathbf{Y} \in \mathbb{R}^{M \times F}`.
        r (float): The radius.
        batch_x (LongTensor, optional): Batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            node to a specific example. :obj:`batch_x` needs to be sorted.
            (default: :obj:`None`)
        batch_y (LongTensor, optional): Batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^M`, which assigns each
            node to a specific example. :obj:`batch_y` needs to be sorted.
            (default: :obj:`None`)
        max_num_neighbors (int, optional): The maximum number of neighbors to
            return for each element in :obj:`y`.
            If the number of actual neighbors is greater than
            :obj:`max_num_neighbors`, returned neighbors are picked randomly.
            (default: :obj:`32`)
        num_workers (int): Number of workers to use for computation. Has no
            effect in case :obj:`batch_x` or :obj:`batch_y` is not
            :obj:`None`, or the input lies on the GPU. (default: :obj:`1`)
        batch_size (int, optional): The number of examples :math:`B`.
            Automatically calculated if not given. (default: :obj:`None`)
        ignore_same_index (bool, optional): If :obj:`True`, each element in
            :obj:`y` ignores the point in :obj:`x` with the same index.
            (default: :obj:`False`)
        use_triton (bool, optional): If :obj:`True`, use Triton kernels when
            available. (default: :obj:`False`)

    .. code-block:: python

        import torch
        from torch_cluster import radius

        x = torch.Tensor([[-1, -1], [-1, 1], [1, -1], [1, 1]])
        batch_x = torch.tensor([0, 0, 0, 0])
        y = torch.Tensor([[-1, 0], [1, 0]])
        batch_y = torch.tensor([0, 0])
        assign_index = radius(x, y, 1.5, batch_x, batch_y)
    """
    if x.numel() == 0 or y.numel() == 0:
        return torch.empty(2, 0, dtype=torch.long, device=x.device)

    x = x.view(-1, 1) if x.dim() == 1 else x
    y = y.view(-1, 1) if y.dim() == 1 else y
    x, y = x.contiguous(), y.contiguous()

    if batch_size is None:
        batch_size = 1
        if batch_x is not None:
            assert x.size(0) == batch_x.numel()
            batch_size = int(batch_x.max()) + 1
        if batch_y is not None:
            assert y.size(0) == batch_y.numel()
            batch_size = max(batch_size, int(batch_y.max()) + 1)
    assert batch_size > 0

    if (use_triton and x.is_cuda and y.is_cuda
            and x.dtype is not torch.float64 and y.dtype is not torch.float64):
        if importlib.util.find_spec("triton") is None:
            print(
                "Triton is not available. Falling back to general "
                "implementation."
            )
        else:
            from .triton.radius import radius as triton_radius
            return triton_radius(
                x,
                y,
                r,
                batch_x,
                batch_y,
                max_num_neighbors,
                batch_size,
                ignore_same_index,
            )

    ptr_x: Optional[torch.Tensor] = None
    ptr_y: Optional[torch.Tensor] = None

    if batch_size > 1:
        assert batch_x is not None
        assert batch_y is not None
        arange = torch.arange(batch_size + 1, device=x.device)
        ptr_x = torch.bucketize(arange, batch_x)
        ptr_y = torch.bucketize(arange, batch_y)

    return torch.ops.torch_cluster.radius(x, y, ptr_x, ptr_y, r,
                                          max_num_neighbors, num_workers,
                                          ignore_same_index)


def radius_graph(
    x: torch.Tensor,
    r: float,
    batch: Optional[torch.Tensor] = None,
    loop: bool = False,
    max_num_neighbors: int = 32,
    flow: str = 'source_to_target',
    num_workers: int = 1,
    batch_size: Optional[int] = None,
    use_triton: bool = False,
) -> torch.Tensor:
    r"""Computes graph edges to all points within a given distance.

    Args:
        x (Tensor): Node feature matrix
            :math:`\mathbf{X} \in \mathbb{R}^{N \times F}`.
        r (float): The radius.
        batch (LongTensor, optional): Batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            node to a specific example. :obj:`batch` needs to be sorted.
            (default: :obj:`None`)
        loop (bool, optional): If :obj:`True`, the graph will contain
            self-loops. (default: :obj:`False`)
        max_num_neighbors (int, optional): The maximum number of neighbors to
            return for each element.
            If the number of actual neighbors is greater than
            :obj:`max_num_neighbors`, returned neighbors are picked randomly.
            (default: :obj:`32`)
        flow (string, optional): The flow direction when used in combination
            with message passing (:obj:`"source_to_target"` or
            :obj:`"target_to_source"`). (default: :obj:`"source_to_target"`)
        num_workers (int): Number of workers to use for computation. Has no
            effect in case :obj:`batch` is not :obj:`None`, or the input lies
            on the GPU. (default: :obj:`1`)
        batch_size (int, optional): The number of examples :math:`B`.
            Automatically calculated if not given. (default: :obj:`None`)
        use_triton (bool, optional): If :obj:`True`, use Triton kernels when
            available. (default: :obj:`False`)

    :rtype: :class:`LongTensor`

    .. code-block:: python

        import torch
        from torch_cluster import radius_graph

        x = torch.Tensor([[-1, -1], [-1, 1], [1, -1], [1, 1]])
        batch = torch.tensor([0, 0, 0, 0])
        edge_index = radius_graph(x, r=1.5, batch=batch, loop=False)
    """

    assert flow in ['source_to_target', 'target_to_source']
    edge_index = radius(
        x,
        x,
        r,
        batch,
        batch,
        max_num_neighbors,
        num_workers,
        batch_size,
        not loop,
        use_triton=use_triton,
    )
    if flow == 'source_to_target':
        return edge_index.flip(0).contiguous()
    return edge_index
