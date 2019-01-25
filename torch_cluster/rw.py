import torch

if torch.cuda.is_available():
    import torch_cluster.rw_cuda


def random_walk(row, col, start, walk_length, num_nodes=None):
    num_nodes = row.max().item() + 1 if num_nodes is None else num_nodes
    if row.is_cuda:
        return torch_cluster.rw_cuda.rw(row, col, start, walk_length, 1, 1,
                                        num_nodes)
    else:
        raise NotImplementedError
