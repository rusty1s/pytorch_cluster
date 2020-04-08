from typing import Optional

import torch
import scipy.spatial


@torch.jit.script
def sample(col: torch.Tensor, count: int) -> torch.Tensor:
    if col.size(0) > count:
        col = col[torch.randperm(col.size(0), dtype=torch.long)][:count]
    return col


def radius(x: torch.Tensor, y: torch.Tensor, r: float,
           batch_x: Optional[torch.Tensor] = None,
           batch_y: Optional[torch.Tensor] = None,
           max_num_neighbors: int = 32) -> torch.Tensor:
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
            node to a specific example. (default: :obj:`None`)
        batch_y (LongTensor, optional): Batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^M`, which assigns each
            node to a specific example. (default: :obj:`None`)
        max_num_neighbors (int, optional): The maximum number of neighbors to
            return for each element in :obj:`y`. (default: :obj:`32`)

    .. code-block:: python

        import torch
        from torch_cluster import radius

        x = torch.Tensor([[-1, -1], [-1, 1], [1, -1], [1, 1]])
        batch_x = torch.tensor([0, 0, 0, 0])
        y = torch.Tensor([[-1, 0], [1, 0]])
        batch_y = torch.tensor([0, 0])
        assign_index = radius(x, y, 1.5, batch_x, batch_y)
    """

    x = x.view(-1, 1) if x.dim() == 1 else x
    y = y.view(-1, 1) if y.dim() == 1 else y

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

        return torch.ops.torch_cluster.radius(x, y, ptr_x, ptr_y, r,
                                              max_num_neighbors)
    else:
        if batch_x is None:
            batch_x = x.new_zeros(x.size(0), dtype=torch.long)

        if batch_y is None:
            batch_y = y.new_zeros(y.size(0), dtype=torch.long)

        assert x.dim() == 2 and batch_x.dim() == 1
        assert y.dim() == 2 and batch_y.dim() == 1
        assert x.size(1) == y.size(1)
        assert x.size(0) == batch_x.size(0)
        assert y.size(0) == batch_y.size(0)

        x = torch.cat([x, 2 * r * batch_x.view(-1, 1).to(x.dtype)], dim=-1)
        y = torch.cat([y, 2 * r * batch_y.view(-1, 1).to(y.dtype)], dim=-1)

        tree = scipy.spatial.cKDTree(x.detach().numpy())
        col = tree.query_ball_point(y.detach().numpy(), r)
        col = [torch.tensor(c, dtype=torch.long) for c in col]
        col = [sample(c, max_num_neighbors) for c in col]
        row = [torch.full_like(c, i) for i, c in enumerate(col)]
        row, col = torch.cat(row, dim=0), torch.cat(col, dim=0)
        mask = col < int(tree.n)
        return torch.stack([row[mask], col[mask]], dim=0)


def radius_graph(x: torch.Tensor, r: float,
                 batch: Optional[torch.Tensor] = None, loop: bool = False,
                 max_num_neighbors: int = 32,
                 flow: str = 'source_to_target') -> torch.Tensor:
    r"""Computes graph edges to all points within a given distance.

    Args:
        x (Tensor): Node feature matrix
            :math:`\mathbf{X} \in \mathbb{R}^{N \times F}`.
        r (float): The radius.
        batch (LongTensor, optional): Batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            node to a specific example. (default: :obj:`None`)
        loop (bool, optional): If :obj:`True`, the graph will contain
            self-loops. (default: :obj:`False`)
        max_num_neighbors (int, optional): The maximum number of neighbors to
            return for each element in :obj:`y`. (default: :obj:`32`)
        flow (string, optional): The flow direction when using in combination
            with message passing (:obj:`"source_to_target"` or
            :obj:`"target_to_source"`). (default: :obj:`"source_to_target"`)

    :rtype: :class:`LongTensor`

    .. code-block:: python

        import torch
        from torch_cluster import radius_graph

        x = torch.Tensor([[-1, -1], [-1, 1], [1, -1], [1, 1]])
        batch = torch.tensor([0, 0, 0, 0])
        edge_index = radius_graph(x, r=1.5, batch=batch, loop=False)
    """

    assert flow in ['source_to_target', 'target_to_source']
    row, col = radius(x, x, r, batch, batch,
                      max_num_neighbors if loop else max_num_neighbors + 1)
    row, col = (col, row) if flow == 'source_to_target' else (row, col)
    if not loop:
        mask = row != col
        row, col = row[mask], col[mask]
    return torch.stack([row, col], dim=0)
