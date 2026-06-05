from __future__ import annotations


import triton
import triton.language as tl
from triton.language.extra import libdevice

from triton.language import topk as tl_topk
import inspect
sig = inspect.signature(tl_topk)
if 'descending' not in sig.parameters:
    from triton.language.standard import sort_impl as tl_topk


def _get_block_d(D, BLOCK_D: int = 16):
    return min(triton.next_power_of_2(D), BLOCK_D)


@triton.jit
def _neumaier_add(acc, acc_c, term):
    t = acc + term
    cond = tl.abs(acc) >= tl.abs(term)
    acc_c += tl.where(
        cond,
        (acc - t) + term,
        (term - t) + acc,
    )
    return t, acc_c


@triton.jit
def _load_tile(ptr, even: tl.constexpr):
    if even:
        return tl.load(ptr)
    return tl.load(
        ptr,
        boundary_check=(0, 1),
        padding_option="zero",
    )


@triton.jit
def _compute_mask_x(
    x_block_start,
    offs_n,
    x_end,
    N,
    mask_y,
    BLOCK_N: tl.constexpr,
):
    x_idx = x_block_start + offs_n
    full_block = (
        (x_block_start + BLOCK_N <= x_end)
        & (x_block_start + BLOCK_N <= N)
        & mask_y
    )
    mask_x = tl.where(
        full_block,
        mask_y,
        (x_idx < x_end) & (x_idx < N) & mask_y,
    )
    return x_idx, full_block, mask_x


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32},
                      num_warps=2,
                      num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64},
                      num_warps=2,
                      num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32},
                      num_warps=2,
                      num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64},
                      num_warps=4,
                      num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64},
                      num_warps=4,
                      num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32},
                      num_warps=4,
                      num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128},
                      num_warps=4,
                      num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128},
                      num_warps=4,
                      num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128},
                      num_warps=8,
                      num_stages=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64},
                      num_warps=8,
                      num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256},
                      num_warps=8,
                      num_stages=2),
    ],
    key=['M', 'N', 'D'],
)
@triton.heuristics({
    'BLOCK_D': lambda args: _get_block_d(args['D']),
    'EVEN_D': lambda args: args['D'] % _get_block_d(args['D']) == 0,
    'EVEN_N': lambda args: args['N'] % args['BLOCK_N'] == 0,
    'EVEN_K': lambda args: args['D'] % args['BLOCK_K'] == 0,
})
@triton.jit
def _nearest_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    M,
    N,
    D,
    stride_xm,
    stride_xd,
    stride_ym,
    stride_yd,
    ptr_y_ptr,
    batch_x_ptr,
    USE_BATCH: tl.constexpr,
    COSINE: tl.constexpr,
    EPS: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_D: tl.constexpr,
):
    r"""Compute nearest neighbor indices for blocks of x.

    Args:
        x_ptr: Pointer to x matrix.
        y_ptr: Pointer to y matrix.
        out_ptr: Pointer to output indices.
        M (int): Number of y rows.
        N (int): Number of x rows.
        D (int): Feature dimension.
        stride_xm/stride_xd: Strides for x.
        stride_ym/stride_yd: Strides for y.
        ptr_y_ptr: Pointer to y batch prefix sums.
        batch_x_ptr: Pointer to x batch ids.
        USE_BATCH (constexpr bool): Whether to apply batch masking.
        COSINE (constexpr bool): Use cosine distance if True.
        EPS (constexpr float): Numerical epsilon.
        BLOCK_M/BLOCK_N/BLOCK_K (constexpr int): Tile sizes.
        EVEN_N (constexpr bool): Alignment hints.
    """
    # X block id.
    pid = tl.program_id(0)
    # X indices.
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    # Valid x rows.
    mask_x = offs_n < N
    if EVEN_N:
        # Hint contiguous access.
        tl.multiple_of(offs_n, 8)
        # Hint vectorization.
        tl.max_contiguous(offs_n, 8)

    # Best dist.
    best_dist = tl.full((BLOCK_N,), float('inf'), tl.float32)
    # Best y index.
    best_idx = tl.zeros((BLOCK_N,), dtype=tl.int32)

    if USE_BATCH:
        # Batch id per x.
        batch_id = tl.load(
            batch_x_ptr + offs_n,
            mask=mask_x,
            other=0,
        )
        left = tl.load(
            ptr_y_ptr + batch_id,
            mask=mask_x,
            other=0,
        )
        # y range start.
        right = tl.load(
            ptr_y_ptr + batch_id + 1,
            mask=mask_x,
            other=0,
        )
        # y range end.
    else:
        # Full y range.
        left = 0
        right = M

    # ||x||^2.
    x_sq = tl.zeros((BLOCK_N,), dtype=tl.float32)
    x_sq_c = tl.zeros((BLOCK_N,), dtype=tl.float32)
    x_block_ptr_sq = tl.make_block_ptr(
        base=x_ptr,
        shape=(D, N),
        strides=(stride_xd, stride_xm),
        offsets=(0, pid * BLOCK_N),
        block_shape=(BLOCK_D, BLOCK_N),
        order=(0, 1),
    )
    for _ in range(0, D, BLOCK_D):
        x = _load_tile(x_block_ptr_sq, EVEN_N and EVEN_D)
        x_term = tl.sum(x * x, axis=0)
        x_sq, x_sq_c = _neumaier_add(x_sq, x_sq_c, x_term)
        x_block_ptr_sq = tl.advance(x_block_ptr_sq, (BLOCK_D, 0))
    x_sq += x_sq_c
    if COSINE:
        # 1/||x||.
        inv_x = tl.rsqrt(x_sq + EPS)

    x_block_ptr_base = tl.make_block_ptr(
        base=x_ptr,
        shape=(D, N),
        strides=(stride_xd, stride_xm),
        offsets=(0, pid * BLOCK_N),
        block_shape=(BLOCK_D, BLOCK_N),
        order=(0, 1),
    )

    for y_start in range(0, M, BLOCK_M):
        # y indices.
        offs_m = y_start + tl.arange(0, BLOCK_M)
        # Valid y rows.
        mask_y = offs_m < M
        # Full tile.
        full_y = y_start + BLOCK_M <= M

        # Dot acc.
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        acc_c = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        # ||y||^2.
        y_sq = tl.zeros((BLOCK_M,), dtype=tl.float32)
        y_sq_c = tl.zeros((BLOCK_M,), dtype=tl.float32)

        x_block_ptr = x_block_ptr_base
        y_block_ptr = tl.make_block_ptr(
            base=y_ptr,
            shape=(M, D),
            strides=(stride_ym, stride_yd),
            offsets=(y_start, 0),
            block_shape=(BLOCK_M, BLOCK_D),
            order=(1, 0),
        )

        for _ in range(0, D, BLOCK_D):
            full_tile = full_y and EVEN_D
            y = _load_tile(y_block_ptr, full_tile)

            full_x = full_tile & EVEN_N
            x = _load_tile(x_block_ptr, full_x)
            dot_term = tl.dot(y, x, input_precision=INPUT_PRECISION)
            acc, acc_c = _neumaier_add(acc, acc_c, dot_term)
            y_term = tl.sum(y * y, axis=1)
            y_sq, y_sq_c = _neumaier_add(y_sq, y_sq_c, y_term)
            y_block_ptr = tl.advance(y_block_ptr, (0, BLOCK_D))
            x_block_ptr = tl.advance(x_block_ptr, (BLOCK_D, 0))

        acc += acc_c
        y_sq += y_sq_c
        if COSINE:
            # 1/||y||.
            inv_y = tl.rsqrt(y_sq + EPS)
            dist = 1.0 - acc * (inv_y[:, None] * inv_x[None, :])
        else:
            dist = (
                (y_sq)[:, None]
                + (x_sq)[None, :]
                - 2.0 * acc
            )

        if full_y:
            valid = tl.broadcast_to(mask_x[None, :], (BLOCK_M, BLOCK_N))
        else:
            valid = mask_y[:, None] & mask_x[None, :]
        if USE_BATCH:
            valid &= (
                (offs_m[:, None] >= left[None, :])
                & (offs_m[:, None] < right[None, :])
            )
        dist = tl.where(valid, dist, float('inf'))

        block_min, block_arg = tl.min(
            dist,
            axis=0,
            return_indices=True,
            return_indices_tie_break_left=True,
        )
        block_idx = y_start + block_arg
        better = block_min < best_dist
        best_dist = tl.where(better, block_min, best_dist)
        best_idx = tl.where(better, block_idx, best_idx)

    # Write output.
    tl.store(
        out_ptr + offs_n,
        best_idx.to(tl.int64),
        mask=mask_x,
    )


@triton.autotune(
    configs=[
        triton.Config(
            {'BLOCK_N': 32},
            num_warps=2,
            num_stages=2,
        ),
        triton.Config(
            {'BLOCK_N': 64},
            num_warps=2,
            num_stages=2,
        ),
        triton.Config(
            {'BLOCK_N': 64},
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {'BLOCK_N': 128},
            num_warps=8,
            num_stages=4,
        ),
        triton.Config(
            {'BLOCK_N': 256},
            num_warps=8,
            num_stages=4,
        ),
    ],
    key=['D', 'MAX_CAND'],
)
@triton.heuristics({
    'BLOCK_D': lambda args: _get_block_d(args['D']),
    'EVEN_D': lambda args: args['D'] % _get_block_d(args['D']) == 0,
    'N_BLOCKS_PER_K': (
        lambda args: triton.cdiv(args['K'], args['BLOCK_N'])
    ),
})
@triton.jit
def _knn_segmented_kernel(
    x_ptr,
    y_ptr,
    ptr_x_ptr,
    batch_y_ptr,
    grid_ptr,
    M,
    N,
    D,
    MAX_CAND,
    stride_xm,
    stride_xd,
    stride_ym,
    stride_yd,
    K: tl.constexpr,
    USE_BATCH: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    EVEN_D: tl.constexpr,
    COSINE: tl.constexpr,
    EPS: tl.constexpr,
    IGNORE_SAME_INDEX: tl.constexpr,
    N_BLOCKS_PER_K: tl.constexpr,
):
    # Program id over y rows.
    pid = tl.program_id(0)
    # Current y index.
    n_y = pid
    # Mask for valid y.
    mask_y = n_y < M

    # Offsets for padded top-k buffer.
    k_offsets = tl.arange(0, N_BLOCKS_PER_K * BLOCK_N)
    INF_KEY = 0x7f800000ffffffff

    best_dist_key = tl.full((N_BLOCKS_PER_K * BLOCK_N,), INF_KEY, tl.int64)

    if N_BLOCKS_PER_K > 1:
        acc_key = tl.full((N_BLOCKS_PER_K, BLOCK_N), INF_KEY, tl.int64)
        k_rows = tl.arange(0, N_BLOCKS_PER_K)[:, None].broadcast_to(
            (N_BLOCKS_PER_K, BLOCK_N)
        )

    if USE_BATCH:
        # Segment id.
        example_idx = tl.load(batch_y_ptr + n_y, mask=mask_y, other=0)
    else:
        example_idx = 0

    # Segment start.
    x_start = tl.load(ptr_x_ptr + example_idx, mask=mask_y, other=0)
    # Segment end.
    x_end = tl.load(ptr_x_ptr + example_idx + 1, mask=mask_y, other=0)
    if COSINE:
        # ||y||^2 accumulator.
        y_sq = tl.zeros((1,), tl.float32)
        y_sq_c = tl.zeros((1,), tl.float32)
        y_norm_ptr = tl.make_block_ptr(
            base=y_ptr,
            shape=(D, M),
            strides=(stride_yd, stride_ym),
            offsets=(0, n_y),
            block_shape=(BLOCK_D, 1),
            order=(0, 1),
        )
        for yi in range(0, D, BLOCK_D):
            y = _load_tile(y_norm_ptr, EVEN_D)
            y_term = tl.sum(y * y, axis=0)
            y_sq, y_sq_c = _neumaier_add(y_sq, y_sq_c, y_term)
            y_norm_ptr = tl.advance(y_norm_ptr, (BLOCK_D, 0))
        # 1/||y|| for cosine.
        y_rnorm = tl.rsqrt(y_sq + EPS)

    for xb in range(0, MAX_CAND, BLOCK_N):
        # Start of x block.
        x_block_start = x_start + xb
        # Offsets within block.
        offs_n = tl.arange(0, BLOCK_N)
        x_idx, full_block, mask_x = _compute_mask_x(
            x_block_start,
            offs_n,
            x_end,
            N,
            mask_y,
            BLOCK_N,
        )
        if MAX_CAND > BLOCK_N:
            # Hint alignment for block start.
            tl.multiple_of(x_block_start, 8)
        # Hint vectorization.
        tl.multiple_of(offs_n, 8)
        tl.max_contiguous(offs_n, 8)

        if COSINE:
            # Dot accumulator.
            acc_dot = tl.zeros((BLOCK_N,), tl.float32)
            acc_dot_c = tl.zeros((BLOCK_N,), tl.float32)
            # ||x||^2 accumulator.
            acc_x_sq = tl.zeros((BLOCK_N,), tl.float32)
            acc_x_sq_c = tl.zeros((BLOCK_N,), tl.float32)
        else:
            acc_dist = tl.zeros((BLOCK_N,), tl.float32)
            acc_dist_c = tl.zeros((BLOCK_N,), tl.float32)
        # Block ptr needs int32 offsets.
        x_block_start_i32 = x_block_start.to(tl.int32)
        n_y_i32 = n_y.to(tl.int32)
        x_block_ptr = tl.make_block_ptr(
            base=x_ptr,
            shape=(N, D),
            strides=(stride_xm, stride_xd),
            offsets=(x_block_start_i32, 0),
            block_shape=(BLOCK_N, BLOCK_D),
            order=(1, 0),
        )
        if COSINE:
            y_block_ptr = tl.make_block_ptr(
                base=y_ptr,
                shape=(D, M),
                strides=(stride_yd, stride_ym),
                offsets=(0, n_y_i32),
                block_shape=(BLOCK_D, 1),
                order=(0, 1),
            )
        else:
            y_block_ptr = tl.make_block_ptr(
                base=y_ptr,
                shape=(M, D),
                strides=(stride_ym, stride_yd),
                offsets=(n_y_i32, 0),
                block_shape=(1, BLOCK_D),
                order=(0, 1),
            )

        for nd in range(0, D, BLOCK_D):
            x = _load_tile(x_block_ptr, EVEN_D)
            y = _load_tile(y_block_ptr, EVEN_D)
            if COSINE:
                # MxV dot.
                prod = tl.dot(x, y, input_precision=INPUT_PRECISION)
                dot_term = tl.sum(prod, axis=1)
                acc_dot, acc_dot_c = _neumaier_add(
                    acc_dot,
                    acc_dot_c,
                    dot_term,
                )
                x_term = tl.sum(x * x, axis=1)
                acc_x_sq, acc_x_sq_c = _neumaier_add(
                    acc_x_sq,
                    acc_x_sq_c,
                    x_term,
                )
                y_block_ptr = tl.advance(y_block_ptr, (BLOCK_D, 0))
            else:
                diff = x - y
                term = tl.sum(diff * diff, axis=1)
                acc_dist, acc_dist_c = _neumaier_add(
                    acc_dist,
                    acc_dist_c,
                    term,
                )
                y_block_ptr = tl.advance(y_block_ptr, (0, BLOCK_D))
            x_block_ptr = tl.advance(x_block_ptr, (0, BLOCK_D))
        if COSINE:
            # 1/||x||.
            x_rnorm = tl.rsqrt(acc_x_sq + acc_x_sq_c + EPS)
            # Cosine distance.
            dist = 1.0 - acc_dot * (x_rnorm * y_rnorm)
        else:
            # L2^2 distance.
            dist = acc_dist + acc_dist_c

        if not full_block:
            # Mask invalid x.
            dist = tl.where(mask_x, dist, float("inf"))

        if IGNORE_SAME_INDEX:
            same_idx = x_idx == n_y
            if not full_block:
                same_idx = same_idx & mask_x
            dist = tl.where(same_idx, float("inf"), dist)

        dist_bits = dist.to(tl.int32, bitcast=True).to(tl.int64)
        # Pack idx.
        idx_bits = x_idx & 0xFFFFFFFF
        # Lexicographic key.
        key = (dist_bits << 32) | idx_bits
        key = tl.where(libdevice.isinf(dist), INF_KEY, key)

        absorb = False
        if N_BLOCKS_PER_K > 1:
            row = (xb // BLOCK_N) % N_BLOCKS_PER_K
            acc_key = tl.where(k_rows == row, key[None, :], acc_key)
            if row == (N_BLOCKS_PER_K - 1) or xb + BLOCK_N > MAX_CAND:
                absorb = True
                final_key = acc_key.ravel()
                acc_key = tl.full(
                    (N_BLOCKS_PER_K, BLOCK_N),
                    INF_KEY,
                    tl.int64,
                )
        else:
            absorb = True
            final_key = key

        if absorb:
            # Merge buffers.
            combo_key = tl.cat(best_dist_key, final_key, can_reorder=True)
            # Get the best.
            best_dist_key = tl_topk(
                combo_key,
                N_BLOCKS_PER_K * BLOCK_N,
                descending=False,
            )

    # Mask for real k.
    k_mask = k_offsets < K
    best_dist_key = tl.where(k_mask, best_dist_key, INF_KEY)
    best_idx = (best_dist_key & 0xFFFFFFFF).to(tl.int32)

    # Output offsets.
    out_offsets = n_y * K + k_offsets
    # Valid output mask.
    out_mask = mask_y & k_mask

    tl.store(grid_ptr + out_offsets, n_y, mask=out_mask)
    # Write col idx.
    tl.store(
        grid_ptr + M * K + out_offsets,
        best_idx.to(tl.int64),
        mask=out_mask,
    )


@triton.autotune(
    configs=[
        triton.Config(
            {'BLOCK_N': 32},
            num_warps=2,
            num_stages=2,
        ),
        triton.Config(
            {'BLOCK_N': 64},
            num_warps=4,
            num_stages=2,
        ),
        triton.Config(
            {'BLOCK_N': 64},
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {'BLOCK_N': 128},
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {'BLOCK_N': 256},
            num_warps=4,
            num_stages=3,
        ),
    ],
    key=['N', 'D', 'MAX_CAND'],
)
@triton.heuristics({
    'BLOCK_D': lambda args: _get_block_d(args['D']),
    'EVEN_D': lambda args: args['D'] % _get_block_d(args['D']) == 0,
    'MAX_X_BLOCKS': lambda args: triton.cdiv(
        args['MAX_CAND'],
        args['BLOCK_N'],
    ),
})
@triton.jit
def _radius_segmented_kernel(
    x_ptr,
    y_ptr,
    ptr_x_ptr,
    example_idx_ptr,
    grid_ptr,
    M,
    N,
    D,
    MAX_CAND,
    stride_xm,
    stride_xd,
    stride_ym,
    stride_yd,
    R2,
    MAX_NEIGHBORS,
    USE_BATCH: tl.constexpr,
    IGNORE_SAME_INDEX: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    MAX_X_BLOCKS: tl.constexpr,
    EVEN_D: tl.constexpr,
):
    pid = tl.program_id(0)
    n_y = pid
    mask_y = n_y < M

    if USE_BATCH:
        example_idx = tl.load(example_idx_ptr + n_y, mask=mask_y, other=0)
    else:
        example_idx = 0

    x_start = tl.load(ptr_x_ptr + example_idx, mask=mask_y, other=0)
    x_end = tl.load(ptr_x_ptr + example_idx + 1, mask=mask_y, other=0)

    if stride_xm % 8 == 0:
        tl.multiple_of(stride_xm, 8)
    if stride_xd % 8 == 0:
        tl.multiple_of(stride_xd, 8)
    if stride_ym % 8 == 0:
        tl.multiple_of(stride_ym, 8)
    if stride_yd % 8 == 0:
        tl.multiple_of(stride_yd, 8)

    count = tl.zeros((), dtype=tl.int32)
    max_neighbors = MAX_NEIGHBORS.to(tl.int32)
    n_y_i32 = n_y.to(tl.int32)
    xb = 0
    offs_n = tl.arange(0, BLOCK_N)

    while (xb < MAX_X_BLOCKS) and (count < max_neighbors):
        x_block_start = x_start + xb * BLOCK_N
        x_idx, full_block, mask_x = _compute_mask_x(
            x_block_start,
            offs_n,
            x_end,
            N,
            mask_y,
            BLOCK_N,
        )

        if MAX_X_BLOCKS > 1:
            tl.multiple_of(x_block_start, 8)

        acc_dist = tl.zeros((BLOCK_N,), tl.float32)
        acc_dist_c = tl.zeros((BLOCK_N,), tl.float32)
        x_block_start_i32 = x_block_start.to(tl.int32)
        x_block_ptr = tl.make_block_ptr(
            base=x_ptr,
            shape=(N, D),
            strides=(stride_xm, stride_xd),
            offsets=(x_block_start_i32, 0),
            block_shape=(BLOCK_N, BLOCK_D),
            order=(1, 0),
        )
        y_row_ptr = tl.make_block_ptr(
            base=y_ptr,
            shape=(M, D),
            strides=(stride_ym, stride_yd),
            offsets=(n_y_i32, 0),
            block_shape=(1, BLOCK_D),
            order=(0, 1),
        )

        for nd in range(0, D, BLOCK_D):
            x = _load_tile(x_block_ptr, EVEN_D)
            y = _load_tile(y_row_ptr, EVEN_D)

            diff = x - y
            term = tl.sum(diff * diff, axis=1)
            acc_dist, acc_dist_c = _neumaier_add(acc_dist, acc_dist_c, term)
            x_block_ptr = tl.advance(x_block_ptr, (0, BLOCK_D))
            y_row_ptr = tl.advance(y_row_ptr, (0, BLOCK_D))

        dist = acc_dist + acc_dist_c
        active = count < max_neighbors
        mask = mask_x & (dist < R2) & active
        if IGNORE_SAME_INDEX:
            mask &= x_idx != n_y.to(tl.int64)

        prefix = tl.cumsum(mask, axis=0).to(tl.int32)
        pos = count + prefix - 1
        write = mask & (pos < max_neighbors)
        pos_safe = tl.where(write, pos, 0)
        out_offset = n_y * MAX_NEIGHBORS + pos_safe
        tl.store(grid_ptr + out_offset, n_y, mask=write)
        tl.store(grid_ptr + M * MAX_NEIGHBORS + out_offset, x_idx, mask=write)
        count += tl.sum(write, axis=0).to(tl.int32)
        xb += 1
