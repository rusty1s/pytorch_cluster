import torch

from .utils import get_func


def grid_cluster(position, size, batch=None):
    # TODO: Check types and sizes

    if batch is not None:
        batch = batch.type_as(position)
        size = torch.cat([size.new(1).fill_(1), size], dim=0)
        dim = position.dim()
        position = torch.cat([batch.unsqueeze(dim - 1), position], dim=dim - 1)

    dim = position.dim()

    # Allow one-dimensional positions.
    if dim == 1:
        position = position.unsqueeze(1)
        dim += 1

    # Translate to minimal positive positions.
    min = position.min(dim=dim - 2, keepdim=True)[0]
    position = position - min

    # Compute cluster count for each dimension.
    max = position.max(dim=0)[0]
    while max.dim() > 1:
        max = max.max(dim=0)[0]
    c_max = torch.ceil(max / size.type_as(max)).long()
    c_max = torch.clamp(c_max, min=1)
    C = c_max.prod()

    # Generate cluster tensor.
    s = list(position.size())
    s[-1] = 1
    cluster = c_max.new(torch.Size(s))

    # Fill cluster tensor and reshape.
    func = get_func('grid', position)
    func(C, cluster, position, size, c_max)
    cluster = cluster.squeeze(dim=dim - 1)

    return cluster
