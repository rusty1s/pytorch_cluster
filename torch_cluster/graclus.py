from typing import Optional

import torch


@torch.jit.script
def graclus_cluster(row: torch.Tensor, col: torch.Tensor,
                    weight: Optional[torch.Tensor] = None,
                    num_nodes: Optional[int] = None) -> torch.Tensor:
    """A greedy clustering algorithm of picking an unmarked vertex and matching
    it with one its unmarked neighbors (that maximizes its edge weight).

    Args:
        row (LongTensor): Source nodes.
        col (LongTensor): Target nodes.
        weight (Tensor, optional): Edge weights. (default: :obj:`None`)
        num_nodes (int, optional): The number of nodes. (default: :obj:`None`)

    :rtype: :class:`LongTensor`

    .. code-block:: python

        import torch
        from torch_cluster import graclus_cluster

        row = torch.tensor([0, 1, 1, 2])
        col = torch.tensor([1, 0, 2, 1])
        weight = torch.Tensor([1, 1, 1, 1])
        cluster = graclus_cluster(row, col, weight)
    """

    if num_nodes is None:
        num_nodes = max(int(row.max()), int(col.max())) + 1

    # Remove self-loops.
    mask = row != col
    row, col = row[mask], col[mask]

    if weight is not None:
        weight = weight[mask]

    # Randomly shuffle nodes.
    if weight is None:
        perm = torch.randperm(row.size(0), dtype=torch.long, device=row.device)
        row, col = row[perm], col[perm]

    # To CSR.
    perm = torch.argsort(row)
    row, col = row[perm], col[perm]

    if weight is not None:
        weight = weight[perm]

    deg = row.new_zeros(num_nodes)
    deg.scatter_add_(0, row, torch.ones_like(row))
    rowptr = row.new_zeros(num_nodes + 1)
    torch.cumsum(deg, 0, out=rowptr[1:])

    return torch.ops.torch_cluster.graclus(rowptr, col, weight)
