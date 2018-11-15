import torch

if torch.cuda.is_available():
    from sample_cuda import farthest_point_sampling, query_radius, query_knn


def batch_slices(batch, sizes=False, include_ends=True):
    """
    Calculates size, start and end indices for each element in a batch.
    """
    batch_size = batch.max().item() + 1
    size = batch.new_zeros(batch_size).scatter_add_(0, batch,
                                                    torch.ones_like(batch))
    cumsum = torch.cumsum(size, dim=0)
    starts = cumsum - size
    ends = cumsum - 1

    slices = starts
    if include_ends:
        slices = torch.stack([starts, ends], dim=1).view(-1)

    if sizes:
        return slices, size
    return slices


def sample_farthest(batch, pos, num_sampled, random_start=False, index=False):
    """Samples a specified number of points for each element in a batch using
    farthest iterative point sampling and returns a mask (or indices) for the
    sampled points. If there are less than num_sampled points in a point cloud
    all points are returned.
    """
    if not pos.is_cuda or not batch.is_cuda:
        raise NotImplementedError

    assert pos.is_contiguous() and batch.is_contiguous()

    slices, sizes = batch_slices(batch, sizes=True)
    batch_size = batch.max().item() + 1

    if random_start:
        random = torch.rand(batch_size, device=slices.device)
        start_points = (sizes.float() * random).long()
    else:
        start_points = torch.zeros_like(sizes)

    idx = farthest_point_sampling(batch_size, slices, pos, num_sampled,
                                  start_points)
    # Remove invalid indices
    idx = idx[idx != -1]

    if index:
        return idx
    mask = torch.zeros(pos.size(0), dtype=torch.uint8, device=pos.device)
    mask[idx] = 1
    return mask


def radius_query_edges(batch,
                       pos,
                       query_batch,
                       query_pos,
                       radius,
                       max_num_neighbors=128,
                       include_self=True):
    if not pos.is_cuda:
        raise NotImplementedError
    assert pos.is_cuda and batch.is_cuda
    assert query_pos.is_cuda and query_batch.is_cuda
    assert pos.is_contiguous() and batch.is_contiguous()
    assert query_pos.is_contiguous() and query_batch.is_contiguous()

    slices, sizes = batch_slices(batch, sizes=True)
    batch_size = batch.max().item() + 1
    query_slices = batch_slices(query_batch)

    max_num_neighbors = min(max_num_neighbors, sizes.max().item())
    idx, cnt = query_radius(batch_size, slices, query_slices, pos, query_pos,
                            radius, max_num_neighbors, include_self)

    # Convert to edges
    view = idx.view(-1)
    row = torch.arange(query_pos.size(0), dtype=torch.long, device=pos.device)
    row = row.view(-1, 1).repeat(1, max_num_neighbors).view(-1)

    # Remove invalid indices
    row = row[view != -1]
    col = view[view != -1]
    if col.size(0) == 0:
        return col

    edge_index = torch.stack([row, col], dim=0)
    return edge_index


def radius_graph(batch, pos, radius, max_num_neighbors=128,
                 include_self=False):
    return radius_query_edges(batch, pos, batch, pos, radius,
                              max_num_neighbors, include_self)


def knn_query_edges(batch,
                    pos,
                    query_batch,
                    query_pos,
                    num_neighbors,
                    include_self=True):
    if not pos.is_cuda:
        raise NotImplementedError
    assert pos.is_cuda and batch.is_cuda
    assert query_pos.is_cuda and query_batch.is_cuda
    assert pos.is_contiguous() and batch.is_contiguous()
    assert query_pos.is_contiguous() and query_batch.is_contiguous()

    slices, sizes = batch_slices(batch, sizes=True)
    batch_size = batch.max().item() + 1
    query_slices = batch_slices(query_batch)

    assert (sizes < num_neighbors).sum().item() == 0

    idx, dists = query_knn(batch_size, slices, query_slices, pos, query_pos,
                           num_neighbors, include_self)

    # Convert to edges
    view = idx.view(-1)

    row = torch.arange(query_pos.size(0), dtype=torch.long, device=pos.device)
    row = row.view(-1, 1).repeat(1, num_neighbors).view(-1)

    # Remove invalid indices
    row = row[view != -1]
    col = view[view != -1]

    edge_index = torch.stack([row, col], dim=0)
    return edge_index


def knn_graph(batch, pos, num_neighbors, include_self=False):
    return knn_query_edges(batch, pos, batch, pos, num_neighbors, include_self)
