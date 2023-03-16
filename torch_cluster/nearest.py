from typing import Optional

import scipy.cluster
import torch


def nearest(
    x: torch.Tensor,
    y: torch.Tensor,
    batch_x: Optional[torch.Tensor] = None,
    batch_y: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""Clusters points in :obj:`x` together which are nearest to a given query
    point in :obj:`y`.

    Args:
        x (Tensor): Node feature matrix
            :math:`\mathbf{X} \in \mathbb{R}^{N \times F}`.
        y (Tensor): Node feature matrix
            :math:`\mathbf{Y} \in \mathbb{R}^{M \times F}`.
        batch_x (LongTensor, optional): Batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            node to a specific example. :obj:`batch_x` needs to be sorted.
            (default: :obj:`None`)
        batch_y (LongTensor, optional): Batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^M`, which assigns each
            node to a specific example. :obj:`batch_y` needs to be sorted.
            (default: :obj:`None`)

    :rtype: :class:`LongTensor`

    .. code-block:: python

        import torch
        from torch_cluster import nearest

        x = torch.Tensor([[-1, -1], [-1, 1], [1, -1], [1, 1]])
        batch_x = torch.tensor([0, 0, 0, 0])
        y = torch.Tensor([[-1, 0], [1, 0]])
        batch_y = torch.tensor([0, 0])
        cluster = nearest(x, y, batch_x, batch_y)
    """

    x = x.view(-1, 1) if x.dim() == 1 else x
    y = y.view(-1, 1) if y.dim() == 1 else y
    assert x.size(1) == y.size(1)

    if batch_x is not None and (batch_x[1:] - batch_x[:-1] < 0).any():
        raise ValueError("'batch_x' is not sorted")
    if batch_y is not None and (batch_y[1:] - batch_y[:-1] < 0).any():
        raise ValueError("'batch_y' is not sorted")

    if x.is_cuda:
        if batch_x is not None:
            assert x.size(0) == batch_x.numel()
            batch_size = int(batch_x.max()) + 1

            deg = x.new_zeros(batch_size, dtype=torch.long)
            deg.scatter_add_(0, batch_x, torch.ones_like(batch_x))

            ptr_x = deg.new_zeros(batch_size + 1)
            torch.cumsum(deg, 0, out=ptr_x[1:])
        else:
            ptr_x = torch.tensor([0, x.size(0)], device=x.device)

        if batch_y is not None:
            assert y.size(0) == batch_y.numel()
            batch_size = int(batch_y.max()) + 1

            deg = y.new_zeros(batch_size, dtype=torch.long)
            deg.scatter_add_(0, batch_y, torch.ones_like(batch_y))

            ptr_y = deg.new_zeros(batch_size + 1)
            torch.cumsum(deg, 0, out=ptr_y[1:])
        else:
            ptr_y = torch.tensor([0, y.size(0)], device=y.device)

        # If an instance in `batch_x` is non-empty, it must be non-empty in
        # `batch_y `as well:
        nonempty_ptr_x = (ptr_x[1:] - ptr_x[:-1]) > 0
        nonempty_ptr_y = (ptr_y[1:] - ptr_y[:-1]) > 0
        if not torch.equal(nonempty_ptr_x, nonempty_ptr_y):
            raise ValueError("Some batch indices occur in 'batch_x' "
                             "that do not occur in 'batch_y'")

        return torch.ops.torch_cluster.nearest(x, y, ptr_x, ptr_y)

    else:

        if batch_x is None and batch_y is not None:
            batch_x = x.new_zeros(x.size(0), dtype=torch.long)
        if batch_y is None and batch_x is not None:
            batch_y = y.new_zeros(y.size(0), dtype=torch.long)

        # Translate and rescale x and y to [0, 1].
        if batch_x is not None and batch_y is not None:
            # If an instance in `batch_x` is non-empty, it must be non-empty in
            # `batch_y `as well:
            unique_batch_x = batch_x.unique_consecutive()
            unique_batch_y = batch_y.unique_consecutive()
            if not torch.equal(unique_batch_x, unique_batch_y):
                raise ValueError("Some batch indices occur in 'batch_x' "
                                 "that do not occur in 'batch_y'")

            assert x.dim() == 2 and batch_x.dim() == 1
            assert y.dim() == 2 and batch_y.dim() == 1
            assert x.size(0) == batch_x.size(0)
            assert y.size(0) == batch_y.size(0)

            min_xy = min(x.min().item(), y.min().item())
            x, y = x - min_xy, y - min_xy

            max_xy = max(x.max().item(), y.max().item())
            x.div_(max_xy)
            y.div_(max_xy)

            # Concat batch/features to ensure no cross-links between examples.
            D = x.size(-1)
            x = torch.cat([x, 2 * D * batch_x.view(-1, 1).to(x.dtype)], -1)
            y = torch.cat([y, 2 * D * batch_y.view(-1, 1).to(y.dtype)], -1)

        return torch.from_numpy(
            scipy.cluster.vq.vq(x.detach().cpu(),
                                y.detach().cpu())[0]).to(torch.long)
