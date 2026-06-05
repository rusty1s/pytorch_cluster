import torch


@torch.library.register_fake("torch_cluster::neighbor_sampler")
def _(start, rowptr, count, factor):
    torch._check(start.device == rowptr.device)
    torch._check(start.ndim == 1)
    torch._check(rowptr.ndim == 1)
    ctx = torch.library.get_ctx()
    out_len = ctx.new_dynamic_size()
    return start.new_empty((out_len,), dtype=torch.long)


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
