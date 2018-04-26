from .utils.loop import remove_self_loops
from .utils.perm import randperm, sort_row, randperm_sort_row
from .utils.ffi import graclus


def graclus_cluster(row, col, weight=None, num_nodes=None):
    """A greedy clustering algorithm of picking an unmarked vertex and matching
    it with one its unmarked neighbors (that maximizes its edge weight).

    Args:
        row (LongTensor): Source nodes.
        col (LongTensor): Target nodes.
        weight (Tensor, optional): Edge weights. (default: :obj:`None`)
        num_nodes (int, optional): The number of nodes. (default: :obj:`None`)

    Examples::

        >>> row = torch.LongTensor([0, 1, 1, 2])
        >>> col = torch.LongTensor([1, 0, 2, 1])
        >>> weight = torch.Tensor([1, 1, 1, 1])
        >>> cluster = graclus_cluster(row, col, weight)
    """

    num_nodes = row.max().item() + 1 if num_nodes is None else num_nodes

    if row.is_cuda:  # pragma: no cover
        row, col = sort_row(row, col)
    else:
        row, col = randperm(row, col)
        row, col = randperm_sort_row(row, col, num_nodes)

    row, col = remove_self_loops(row, col)
    cluster = row.new_empty((num_nodes, ))
    graclus(cluster, row, col, weight)

    return cluster
