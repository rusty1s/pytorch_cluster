from .utils.ffi import grid


def grid_cluster(pos, size, start=None, end=None):
    pos = pos.unsqueeze(-1) if pos.dim() == 1 else pos

    assert pos.size(1) == size.size(0), (
        'Last dimension of position tensor must have same size as size tensor')

    start = pos.t().min(dim=1)[0] if start is None else start
    end = pos.t().max(dim=1)[0] if end is None else end
    pos, end = pos - start, end - start

    size = size.type_as(pos)
    count = (end / size).long() + 1

    cluster = count.new(pos.size(0))
    grid(cluster, pos, size, count)

    return cluster
