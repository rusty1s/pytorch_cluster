from typing import Optional

import torch


@torch.jit.script
def grid_cluster(pos: torch.Tensor, size: torch.Tensor,
                 start: Optional[torch.Tensor] = None,
                 end: Optional[torch.Tensor] = None) -> torch.Tensor:
    """A clustering algorithm, which overlays a regular grid of user-defined
    size over a point cloud and clusters all points within a voxel.

    Args:
        pos (Tensor): D-dimensional position of points.
        size (Tensor): Size of a voxel in each dimension.
        start (Tensor, optional): Start position of the grid (in each
            dimension). (default: :obj:`None`)
        end (Tensor, optional): End position of the grid (in each
            dimension). (default: :obj:`None`)

    :rtype: :class:`LongTensor`

    .. code-block:: python

        import torch
        from torch_cluster import grid_cluster

        pos = torch.Tensor([[0, 0], [11, 9], [2, 8], [2, 2], [8, 3]])
        size = torch.Tensor([5, 5])
        cluster = grid_cluster(pos, size)
    """
    return torch.ops.torch_cluster.grid(pos, size, start, end)
