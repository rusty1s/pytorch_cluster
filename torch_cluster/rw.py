import warnings

import torch

if torch.cuda.is_available():
    import torch_cluster.rw_cuda


def random_walk(row, col, start, walk_length, p=1, q=1, num_nodes=None):
    if p != 1 or q != 1:
        warnings.warn('Parameters `p` and `q` are not supported yet and will'
                      'be restored to their default values `p=1` and `q=1`.')
        p = q = 1

    num_nodes = row.max().item() + 1 if num_nodes is None else num_nodes
    if row.is_cuda:
        return torch_cluster.rw_cuda.rw(row, col, start, walk_length, p, q,
                                        num_nodes)
    else:
        raise NotImplementedError
