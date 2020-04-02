import warnings
from typing import Optional

import torch


@torch.jit.script
def random_walk(row: torch.Tensor, col: torch.Tensor, start: torch.Tensor,
                walk_length: int, p: float = 1, q: float = 1,
                coalesced: bool = True, num_nodes: Optional[int] = None):
    """Samples random walks of length :obj:`walk_length` from all node indices
    in :obj:`start` in the graph given by :obj:`(row, col)` as described in the
    `"node2vec: Scalable Feature Learning for Networks"
    <https://arxiv.org/abs/1607.00653>`_ paper.
    Edge indices :obj:`(row, col)` need to be coalesced/sorted according to
    :obj:`row` (use the :obj:`coalesced` attribute to force).

    Args:
        row (LongTensor): Source nodes.
        col (LongTensor): Target nodes.
        start (LongTensor): Nodes from where random walks start.
        walk_length (int): The walk length.
        p (float, optional): Likelihood of immediately revisiting a node in the
            walk. (default: :obj:`1`)
        q (float, optional): Control parameter to interpolate between
            breadth-first strategy and depth-first strategy (default: :obj:`1`)
        coalesced (bool, optional): If set to :obj:`True`, will coalesce/sort
            the graph given by :obj:`(row, col)` according to :obj:`row`.
            (default: :obj:`True`)
        num_nodes (int, optional): The number of nodes. (default: :obj:`None`)

    :rtype: :class:`LongTensor`
    """
    if num_nodes is None:
        num_nodes = max(int(row.max()), int(col.max())) + 1

    if coalesced:
        perm = torch.argsort(row * num_nodes + col)
        row, col = row[perm], col[perm]

    deg = row.new_zeros(num_nodes)
    deg.scatter_add_(0, row, torch.ones_like(row))
    rowptr = row.new_zeros(num_nodes + 1)
    torch.cumsum(deg, 0, out=rowptr[1:])

    if p != 1. or q != 1.:  # pragma: no cover
        warnings.warn('Parameters `p` and `q` are not supported yet and will'
                      'be restored to their default values `p=1` and `q=1`.')
        p = q = 1.

    return torch.ops.torch_cluster.random_walk(rowptr, col, start, walk_length,
                                               p, q)
