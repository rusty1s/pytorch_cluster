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
    p_min = position.min(dim=-2, keepdim=True)[0]
    position = position - p_min

    # Compute maximal position for each dimension.
    p_max = position.max(dim=0)[0]
    while p_max.dim() > 1:
        p_max = p_max.max(dim=0)[0]

    # Generate cluster tensor.
    s = list(position.size())[:-1] + [1]
    cluster = size.new(torch.Size(s)).long()

    # Fill cluster tensor and reshape.
    size = size.type_as(position)
    func = get_func('grid', position)
    C = func(cluster, position, size, p_max)
    cluster = cluster.squeeze(dim=-1)
    cluster, u = consecutive(cluster)

    if batch is None:
        return cluster, None
    else:
        print(p_max.tolist(), size.tolist(), C)
        batch = (u / C).long()
        return cluster, batch
