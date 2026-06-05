from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from ._kernels import _knn_segmented_kernel


def segmented_topk_search(
    x: Tensor,
    y: Tensor,
    k: int,
    batch_x: Optional[Tensor] = None,
    batch_y: Optional[Tensor] = None,
    cosine: bool = False,
    batch_size: Optional[int] = None,
    ignore_same_index: bool = False,
) -> Tensor:
    r"""Compute top-k neighbor indices."""
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
        batch_y = batch_y.to(torch.int64).contiguous()

    seg_sizes = torch.diff(ptr_x).to(torch.int64)
    if seg_sizes.numel() > 0:
        max_candidates = int(seg_sizes.max().item())
    else:
        max_candidates = 0
    eps = 0.0 if cosine else torch.finfo(torch.float32).eps

    if max_candidates == 0:
        return torch.full((2, M * k), -1, device=y.device, dtype=torch.int64)

    grid_out = torch.empty(2, M * k, device=y.device, dtype=torch.int64)
    grid = (M,)
    _knn_segmented_kernel[grid](
        x,
        y,
        ptr_x,
        batch_y if use_batch else ptr_x,  # dummy ptr when unused
        grid_out,
        M,
        N,
        D,
        max_candidates,
        x.stride(0),
        x.stride(1),
        y.stride(0),
        y.stride(1),
        K=k,
        USE_BATCH=use_batch,
        INPUT_PRECISION="ieee",
        COSINE=cosine,
        EPS=eps,
        IGNORE_SAME_INDEX=ignore_same_index,
    )
    return grid_out
