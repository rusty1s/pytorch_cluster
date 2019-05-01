#include <ATen/ATen.h>

#include "atomics.cuh"
#include "utils.cuh"

#define THREADS 1024

template <typename scalar_t, int64_t Dim> struct Dist;

template <typename scalar_t> struct Dist<scalar_t, 1> {
  static __device__ void
  compute(ptrdiff_t idx, ptrdiff_t start_idx, ptrdiff_t end_idx, ptrdiff_t old,
          scalar_t *__restrict__ best, ptrdiff_t *__restrict__ best_idx,
          const scalar_t *__restrict__ x, scalar_t *__restrict__ dist,
          scalar_t *__restrict__ tmp_dist, size_t dim) {

    for (ptrdiff_t n = start_idx + idx; n < end_idx; n += THREADS) {
      scalar_t d = x[old] - x[n];
      dist[n] = min(dist[n], d * d);
      if (dist[n] > *best) {
        *best = dist[n];
        *best_idx = n;
      }
    }
  }
};

template <typename scalar_t> struct Dist<scalar_t, 2> {
  static __device__ void
  compute(ptrdiff_t idx, ptrdiff_t start_idx, ptrdiff_t end_idx, ptrdiff_t old,
          scalar_t *__restrict__ best, ptrdiff_t *__restrict__ best_idx,
          const scalar_t *__restrict__ x, scalar_t *__restrict__ dist,
          scalar_t *__restrict__ tmp_dist, size_t dim) {

    for (ptrdiff_t n = start_idx + idx; n < end_idx; n += THREADS) {
      scalar_t a = x[2 * old + 0] - x[2 * n + 0];
      scalar_t b = x[2 * old + 1] - x[2 * n + 1];
      scalar_t d = a * a + b * b;
      dist[n] = min(dist[n], d);
      if (dist[n] > *best) {
        *best = dist[n];
        *best_idx = n;
      }
    }
  }
};

template <typename scalar_t> struct Dist<scalar_t, 3> {
  static __device__ void
  compute(ptrdiff_t idx, ptrdiff_t start_idx, ptrdiff_t end_idx, ptrdiff_t old,
          scalar_t *__restrict__ best, ptrdiff_t *__restrict__ best_idx,
          const scalar_t *__restrict__ x, scalar_t *__restrict__ dist,
          scalar_t *__restrict__ tmp_dist, size_t dim) {

    for (ptrdiff_t n = start_idx + idx; n < end_idx; n += THREADS) {
      scalar_t a = x[3 * old + 0] - x[3 * n + 0];
      scalar_t b = x[3 * old + 1] - x[3 * n + 1];
      scalar_t c = x[3 * old + 2] - x[3 * n + 2];
      scalar_t d = a * a + b * b + c * c;
      dist[n] = min(dist[n], d);
      if (dist[n] > *best) {
        *best = dist[n];
        *best_idx = n;
      }
    }
  }
};

template <typename scalar_t> struct Dist<scalar_t, -1> {
  static __device__ void
  compute(ptrdiff_t idx, ptrdiff_t start_idx, ptrdiff_t end_idx, ptrdiff_t old,
          scalar_t *__restrict__ best, ptrdiff_t *__restrict__ best_idx,
          const scalar_t *__restrict__ x, scalar_t *__restrict__ dist,
          scalar_t *__restrict__ tmp_dist, size_t dim) {

    for (ptrdiff_t n = start_idx + idx; n < end_idx; n += THREADS) {
      tmp_dist[n] = 0;
    }

    __syncthreads();
    for (ptrdiff_t i = start_idx * dim + idx; i < end_idx * dim; i += THREADS) {
      scalar_t d = x[(old * dim) + (i % dim)] - x[i];
      atomicAdd(&tmp_dist[i / dim], d * d);
    }

    __syncthreads();
    for (ptrdiff_t n = start_idx + idx; n < end_idx; n += THREADS) {
      dist[n] = min(dist[n], tmp_dist[n]);
      if (dist[n] > *best) {
        *best = dist[n];
        *best_idx = n;
      }
    }
  }
};

template <typename scalar_t, int64_t Dim>
__global__ void
fps_kernel(const scalar_t *__restrict__ x, const int64_t *__restrict__ cum_deg,
           const int64_t *__restrict__ cum_k, const int64_t *__restrict__ start,
           scalar_t *__restrict__ dist, scalar_t *__restrict__ tmp_dist,
           int64_t *__restrict__ out, size_t dim) {

  const ptrdiff_t batch_idx = blockIdx.x;
  const ptrdiff_t idx = threadIdx.x;

  const ptrdiff_t start_idx = cum_deg[batch_idx];
  const ptrdiff_t end_idx = cum_deg[batch_idx + 1];

  __shared__ scalar_t best_dist[THREADS];
  __shared__ int64_t best_dist_idx[THREADS];

  if (idx == 0) {
    out[cum_k[batch_idx]] = start_idx + start[batch_idx];
  }

  for (ptrdiff_t m = cum_k[batch_idx] + 1; m < cum_k[batch_idx + 1]; m++) {
    scalar_t best = -1;
    ptrdiff_t best_idx = 0;

    __syncthreads();
    Dist<scalar_t, Dim>::compute(idx, start_idx, end_idx, out[m - 1], &best,
                                 &best_idx, x, dist, tmp_dist, dim);

    best_dist[idx] = best;
    best_dist_idx[idx] = best_idx;

    for (int64_t u = 0; (1 << u) < THREADS; u++) {
      __syncthreads();
      if (idx < (THREADS >> (u + 1))) {
        int64_t idx_1 = (idx * 2) << u;
        int64_t idx_2 = (idx * 2 + 1) << u;
        if (best_dist[idx_1] < best_dist[idx_2]) {
          best_dist[idx_1] = best_dist[idx_2];
          best_dist_idx[idx_1] = best_dist_idx[idx_2];
        }
      }
    }

    __syncthreads();
    if (idx == 0) {
      out[m] = best_dist_idx[0];
    }
  }
}

#define FPS_KERNEL(DIM, ...)                                                   \
  [&] {                                                                        \
    switch (DIM) {                                                             \
    case 1:                                                                    \
      fps_kernel<scalar_t, 1><<<batch_size, THREADS>>>(__VA_ARGS__, DIM);      \
      break;                                                                   \
    case 2:                                                                    \
      fps_kernel<scalar_t, 2><<<batch_size, THREADS>>>(__VA_ARGS__, DIM);      \
      break;                                                                   \
    case 3:                                                                    \
      fps_kernel<scalar_t, 3><<<batch_size, THREADS>>>(__VA_ARGS__, DIM);      \
      break;                                                                   \
    default:                                                                   \
      fps_kernel<scalar_t, -1><<<batch_size, THREADS>>>(__VA_ARGS__, DIM);     \
    }                                                                          \
  }()

at::Tensor fps_cuda(at::Tensor x, at::Tensor batch, float ratio, bool random) {
  cudaSetDevice(x.get_device());
  auto batch_sizes = (int64_t *)malloc(sizeof(int64_t));
  cudaMemcpy(batch_sizes, batch[-1].data<int64_t>(), sizeof(int64_t),
             cudaMemcpyDeviceToHost);
  auto batch_size = batch_sizes[0] + 1;

  auto deg = degree(batch, batch_size);
  auto cum_deg = at::cat({at::zeros(1, deg.options()), deg.cumsum(0)}, 0);
  auto k = (deg.toType(at::kFloat) * ratio).ceil().toType(at::kLong);
  auto cum_k = at::cat({at::zeros(1, k.options()), k.cumsum(0)}, 0);

  at::Tensor start;
  if (random) {
    start = at::rand(batch_size, x.options());
    start = (start * deg.toType(at::kFloat)).toType(at::kLong);
  } else {
    start = at::zeros(batch_size, k.options());
  }

  auto dist = at::full(x.size(0), 1e38, x.options());
  auto tmp_dist = at::empty(x.size(0), x.options());

  auto k_sum = (int64_t *)malloc(sizeof(int64_t));
  cudaMemcpy(k_sum, cum_k[-1].data<int64_t>(), sizeof(int64_t),
             cudaMemcpyDeviceToHost);
  auto out = at::empty(k_sum[0], k.options());

  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "fps_kernel", [&] {
    FPS_KERNEL(x.size(1), x.data<scalar_t>(), cum_deg.data<int64_t>(),
               cum_k.data<int64_t>(), start.data<int64_t>(),
               dist.data<scalar_t>(), tmp_dist.data<scalar_t>(),
               out.data<int64_t>());
  });

  return out;
}
