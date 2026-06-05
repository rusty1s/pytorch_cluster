from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from ._kernels import _radius_segmented_kernel


def radius(
    x: Tensor,
    y: Tensor,
    r: float,
    batch_x: Optional[Tensor] = None,
    batch_y: Optional[Tensor] = None,
    max_num_neighbors: int = 32,
    batch_size: Optional[int] = None,
    ignore_same_index: bool = False,
) -> Tensor:
    r"""Find all neighbors within a radius using Triton kernels.

    Args:
        x (Tensor): Input features of shape [N, D].
        y (Tensor): Query features of shape [M, D].
        r (float): Search radius.
        batch_x (Tensor, optional): Batch vector for x.
        batch_y (Tensor, optional): Batch vector for y.
        max_num_neighbors (int, optional): Maximum neighbors per y.
        batch_size (int, optional): Number of batches.
        ignore_same_index (bool, optional): Exclude exact self matches.

    Returns:
        Tensor: Edge index with shape [2, num_edges].
    """
    use_batch = (batch_size or 1) > 1
    if use_batch:
        assert batch_x is not None
        assert batch_y is not None
        arange = torch.arange((batch_size or 1) + 1, device=x.device)
        ptr_x = torch.bucketize(arange, batch_x)
        ptr_y = torch.bucketize(arange, batch_y)
    else:
        ptr_x = torch.tensor(
            [0, x.size(0)],
            device=x.device,
            dtype=torch.int64,
        )
        ptr_y = torch.tensor(
            [0, y.size(0)],
            device=y.device,
            dtype=torch.int64,
        )

    if ptr_x.numel() != ptr_y.numel():
        raise ValueError(
            "ptr_x and ptr_y must have the same number of segments."
        )

    M, N = y.size(0), x.size(0)
    D = x.size(1)
    if use_batch:
        example_idx = batch_y.to(torch.int64).contiguous()
    else:
        example_idx = ptr_x  # Dummy; kernel ignores when USE_BATCH=False.

    seg_sizes = torch.diff(ptr_x).to(torch.int64)
    if seg_sizes.numel() > 0:
        max_candidates = int(seg_sizes.max().item())
    else:
        max_candidates = 0

    if max_candidates == 0 or max_num_neighbors <= 0:
        return torch.empty(2, 0, device=x.device, dtype=torch.int64)

    grid_out = torch.full(
        (2, M * max_num_neighbors),
        -1,
        device=x.device,
        dtype=torch.int64,
    )
    grid = (M,)
    _radius_segmented_kernel[grid](
        x,
        y,
        ptr_x,
        example_idx,
        grid_out,
        M,
        N,
        D,
        max_candidates,
        x.stride(0),
        x.stride(1),
        y.stride(0),
        y.stride(1),
        float(r) * float(r),
        max_num_neighbors,
        USE_BATCH=use_batch,
        IGNORE_SAME_INDEX=ignore_same_index,
    )

    return grid_out[:, grid_out[0] != -1]
