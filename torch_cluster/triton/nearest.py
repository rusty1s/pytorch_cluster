from __future__ import annotations

from typing import Optional

import torch
import triton
from torch import Tensor

from ._kernels import _nearest_kernel


def nearest(
    x: Tensor,
    y: Tensor,
    batch_x: Optional[Tensor] = None,
    batch_y: Optional[Tensor] = None,
) -> Tensor:
    r"""Find nearest neighbors using Triton pairwise distances.

    Args:
        x (Tensor): Input features of shape [N, D].
        y (Tensor): Query features of shape [M, D].
        batch_x (Tensor, optional): Batch vector for x.
        batch_y (Tensor, optional): Batch vector for y.

    Returns:
        Tensor: Index of the nearest y for each x.
    """
    if batch_x is None and batch_y is None:
        out = torch.empty((x.size(0),), dtype=torch.long, device=x.device)

        def grid(meta):
            return (triton.cdiv(x.size(0), meta['BLOCK_N']),)

        eps = torch.finfo(torch.float32).eps
        _nearest_kernel[grid](
            x,
            y,
            out,
            y.size(0),
            x.size(0),
            x.size(1),
            x.stride(0),
            x.stride(1),
            y.stride(0),
            y.stride(1),
            y,
            x,
            USE_BATCH=False,
            COSINE=False,
            EPS=eps,
            INPUT_PRECISION="ieee",
        )
        return out

    if batch_x is None:
        batch_x = x.new_zeros(x.size(0), dtype=torch.long)
    if batch_y is None:
        batch_y = y.new_zeros(y.size(0), dtype=torch.long)

    batch_x_unique = batch_x.unique_consecutive()
    batch_y_unique = batch_y.unique_consecutive()
    if not torch.equal(batch_x_unique, batch_y_unique):
        raise ValueError("Some batch indices occur in 'batch_x' "
                         "that do not occur in 'batch_y'")

    batch_size = int(max(batch_x.max(), batch_y.max())) + 1
    use_batch = batch_size > 1
    if use_batch:
        arange = torch.arange(batch_size + 1, device=x.device)
        ptr_y = torch.bucketize(arange, batch_y)
    else:
        ptr_y = y
        batch_x = x
    out = torch.empty((x.size(0),), dtype=torch.long, device=x.device)

    def grid(meta):
        return (triton.cdiv(x.size(0), meta['BLOCK_N']),)

    eps = torch.finfo(torch.float32).eps
    _nearest_kernel[grid](
        x,
        y,
        out,
        y.size(0),
        x.size(0),
        x.size(1),
        x.stride(0),
        x.stride(1),
        y.stride(0),
        y.stride(1),
        ptr_y,
        batch_x,
        USE_BATCH=use_batch,
        COSINE=False,
        EPS=eps,
        INPUT_PRECISION="ieee",
    )
    return out
