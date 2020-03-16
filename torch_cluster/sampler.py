import torch


@torch.jit.script
def neighbor_sampler(start: torch.Tensor, rowptr: torch.Tensor, size: float):
    assert not start.is_cuda

    factor: float = -1.
    count: int = -1
    if size <= 1:
        factor = size
        assert factor > 0
    else:
        count = int(size)

    return torch.ops.torch_cluster.neighbor_sampler(start, rowptr, count,
                                                    factor)
