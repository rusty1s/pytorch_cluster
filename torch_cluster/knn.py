from typing import Optional

import torch
import numpy as np


def knn(x: torch.Tensor, y: torch.Tensor, k: int,
        batch_x: Optional[torch.Tensor] = None,
        batch_y: Optional[torch.Tensor] = None,
        cosine: bool = False, n_threads: int = 1) -> torch.Tensor:
    r"""Finds for each element in :obj:`y` the :obj:`k` nearest points in
    :obj:`x`.

    Args:
        x (Tensor): Node feature matrix
            :math:`\mathbf{X} \in \mathbb{R}^{N \times F}`.
        y (Tensor): Node feature matrix
            :math:`\mathbf{X} \in \mathbb{R}^{M \times F}`.
        k (int): The number of neighbors.
        batch_x (LongTensor, optional): Batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            node to a specific example. (default: :obj:`None`)
        batch_y (LongTensor, optional): Batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^M`, which assigns each
            node to a specific example. (default: :obj:`None`)
        cosine (boolean, optional): If :obj:`True`, will use the cosine
            distance instead of euclidean distance to find nearest neighbors.
            (default: :obj:`False`)

    :rtype: :class:`LongTensor`

    .. code-block:: python

        import torch
        from torch_cluster import knn

        x = torch.Tensor([[-1, -1], [-1, 1], [1, -1], [1, 1]])
        batch_x = torch.tensor([0, 0, 0, 0])
        y = torch.Tensor([[-1, 0], [1, 0]])
        batch_x = torch.tensor([0, 0])
        assign_index = knn(x, y, 2, batch_x, batch_y)
    """

    x = x.view(-1, 1) if x.dim() == 1 else x
    y = y.view(-1, 1) if y.dim() == 1 else y

    def is_sorted(x):
        return (np.diff(x.detach().cpu()) >= 0).all()

    if x.is_cuda:
        if batch_x is not None:
            assert x.size(0) == batch_x.numel()
            assert is_sorted(batch_x)
            batch_size = int(batch_x.max()) + 1

            deg = x.new_zeros(batch_size, dtype=torch.long)
            deg.scatter_add_(0, batch_x, torch.ones_like(batch_x))

            ptr_x = deg.new_zeros(batch_size + 1)
            torch.cumsum(deg, 0, out=ptr_x[1:])
        else:
            ptr_x = torch.tensor([0, x.size(0)], device=x.device)

        if batch_y is not None:
            assert y.size(0) == batch_y.numel()
            assert is_sorted(batch_y)
            batch_size = int(batch_y.max()) + 1

            deg = y.new_zeros(batch_size, dtype=torch.long)
            deg.scatter_add_(0, batch_y, torch.ones_like(batch_y))

            ptr_y = deg.new_zeros(batch_size + 1)
            torch.cumsum(deg, 0, out=ptr_y[1:])
        else:
            ptr_y = torch.tensor([0, y.size(0)], device=y.device)

        return torch.ops.torch_cluster.knn(x, y, ptr_x,
                                           ptr_y, k, cosine, n_threads)
    else:
        assert x.dim() == 2
        if batch_x is not None:
            assert batch_x.dim() == 1
            assert is_sorted(batch_x)
            assert x.size(0) == batch_x.size(0)

        assert y.dim() == 2
        if batch_y is not None:
            assert batch_y.dim() == 1
            assert is_sorted(batch_y)
            assert y.size(0) == batch_y.size(0)
        assert x.size(1) == y.size(1)

        if cosine:
            raise NotImplementedError('`cosine` argument not supported on CPU')

        return torch.ops.torch_cluster.knn(x, y, batch_x, batch_y,
                                           k, cosine, n_threads)


def knn_graph(x: torch.Tensor, k: int, batch: Optional[torch.Tensor] = None,
              loop: bool = False, flow: str = 'source_to_target',
              cosine: bool = False, n_threads: int = 1) -> torch.Tensor:
    r"""Computes graph edges to the nearest :obj:`k` points.

    Args:
        x (Tensor): Node feature matrix
            :math:`\mathbf{X} \in \mathbb{R}^{N \times F}`.
        k (int): The number of neighbors.
        batch (LongTensor, optional): Batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            node to a specific example. (default: :obj:`None`)
        loop (bool, optional): If :obj:`True`, the graph will contain
            self-loops. (default: :obj:`False`)
        flow (string, optional): The flow direction when using in combination
            with message passing (:obj:`"source_to_target"` or
            :obj:`"target_to_source"`). (default: :obj:`"source_to_target"`)
        cosine (boolean, optional): If :obj:`True`, will use the cosine
            distance instead of euclidean distance to find nearest neighbors.
            (default: :obj:`False`)

    :rtype: :class:`LongTensor`

    .. code-block:: python

        import torch
        from torch_cluster import knn_graph

        x = torch.Tensor([[-1, -1], [-1, 1], [1, -1], [1, 1]])
        batch = torch.tensor([0, 0, 0, 0])
        edge_index = knn_graph(x, k=2, batch=batch, loop=False)
    """

    assert flow in ['source_to_target', 'target_to_source']
    row, col = knn(x, x, k if loop else k + 1, batch, batch,
                   cosine=cosine, n_threads=n_threads)
    row, col = (col, row) if flow == 'source_to_target' else (row, col)
    if not loop:
        mask = row != col
        row, col = row[mask], col[mask]
    return torch.stack([row, col], dim=0)
