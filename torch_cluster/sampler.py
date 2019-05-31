import torch_cluster.sampler_cpu


def neighbor_sampler(start, cumdeg, size):
    assert not start.is_cuda

    factor = 1
    if isinstance(size, float):
        factor = size
        size = 2147483647

    op = torch_cluster.sampler_cpu.neighbor_sampler
    return op(start, cumdeg, size, factor)
