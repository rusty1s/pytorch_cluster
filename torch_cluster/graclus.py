import torch
import graclus_cpu

if torch.cuda.is_available():
    import graclus_cuda


def graclus_cluster(row, col, weight=None, num_nodes=None):
    """A greedy clustering algorithm of picking an unmarked vertex and matching
    it with one its unmarked neighbors (that maximizes its edge weight).

    Args:
        row (LongTensor): Source nodes.
        col (LongTensor): Target nodes.
        weight (Tensor, optional): Edge weights. (default: :obj:`None`)
        num_nodes (int, optional): The number of nodes. (default: :obj:`None`)

    :rtype: :class:`LongTensor`

    Examples::

        >>> row = torch.tensor([0, 1, 1, 2])
        >>> col = torch.tensor([1, 0, 2, 1])
        >>> weight = torch.Tensor([1, 1, 1, 1])
        >>> cluster = graclus_cluster(row, col, weight)
    """

    if num_nodes is None:
        num_nodes = max(row.max().item(), col.max().item()) + 1

    op = graclus_cuda if row.is_cuda else graclus_cpu

    if weight is None:
        cluster = op.graclus(row, col, num_nodes)
    else:
        cluster = op.weighted_graclus(row, col, weight, num_nodes)

    return cluster
