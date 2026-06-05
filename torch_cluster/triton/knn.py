from __future__ import annotations
from typing import Optional

from torch import Tensor

from .segmented import segmented_topk_search


def knn(
    x: Tensor,
    y: Tensor,
    k: int,
    batch_x: Optional[Tensor] = None,
    batch_y: Optional[Tensor] = None,
    cosine: bool = False,
    batch_size: Optional[int] = None,
) -> Tensor:
    r"""Compute k-NN indices using Triton kernels.

    Args:
        x (Tensor): Input features of shape [N, D].
        y (Tensor): Query features of shape [M, D].
        k (int): Number of neighbors.
        batch_x (Tensor, optional): Batch vector for x.
        batch_y (Tensor, optional): Batch vector for y.
        cosine (bool, optional): Use cosine distance if True.
        num_workers (int, optional): Unused for Triton path.
        batch_size (int, optional): Number of batches.
        ignore_same_index (bool, optional): Exclude exact self matches.

    Returns:
        Tensor: Edge index with shape [2, M * k].
    """
    grid = segmented_topk_search(
        x,
        y,
        k,
        batch_x,
        batch_y,
        cosine,
        batch_size,
    )
    return grid[:, grid[1] >= 0]
