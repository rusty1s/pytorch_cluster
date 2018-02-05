import torch

from .utils import get_func, consecutive


def grid_cluster(position, size, batch=None):
    # Allow one-dimensional positions.
    if position.dim() == 1:
        position = position.unsqueeze(-1)

    assert size.dim() == 1, 'Size tensor must be one-dimensional'
    assert position.size(-1) == size.size(-1), (
        'Last dimension of position tensor must have same size as size tensor')

    # If given, append batch to position tensor.
    if batch is not None:
        batch = batch.unsqueeze(-1).type_as(position)
        assert position.size()[:-1] == batch.size()[:-1], (
            'Position tensor must have same size as batch tensor apart from '
            'the last dimension')
        position = torch.cat([batch, position], dim=-1)
        size = torch.cat([size.new(1).fill_(1), size], dim=-1)

    # Translate to minimal positive positions.
    min = position.min(dim=-2, keepdim=True)[0]
    position = position - min

    # Compute cluster count for each dimension.
    max = position.max(dim=0)[0]
    while max.dim() > 1:
        max = max.max(dim=0)[0]
    c_max = torch.floor(max.double() / size.double() + 1).long()
    c_max = torch.clamp(c_max, min=1)
    C = c_max.prod()

    # Generate cluster tensor.
    s = list(position.size())
    s[-1] = 1
    cluster = c_max.new(torch.Size(s))

    # Fill cluster tensor and reshape.
    size = size.type_as(position)
    func = get_func('grid', position)
    func(C, cluster, position, size, c_max)
    cluster = cluster.squeeze(dim=-1)
    if not cluster.is_cuda:
        cluster = consecutive(cluster)

    return cluster
