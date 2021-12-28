#include "nearest_cuda.h"

#include <ATen/cuda/CUDAContext.h>

#include "utils.cuh"

#define THREADS 1024

template <typename scalar_t>
__global__ void nearest_kernel(const scalar_t *x, const scalar_t *y,
                               const int64_t *ptr_x, const int64_t *ptr_y,
                               int64_t *out, int64_t batch_size, int64_t dim) {

  const int64_t thread_idx = threadIdx.x;
  const int64_t n_x = blockIdx.x;

  int64_t batch_idx;
  for (int64_t b = 0; b < batch_size; b++) {
    if (n_x >= ptr_x[b] && n_x < ptr_x[b + 1]) {
      batch_idx = b;
      break;
    }
  }

  const int64_t y_start_idx = ptr_y[batch_idx];
  const int64_t y_end_idx = ptr_y[batch_idx + 1];

  __shared__ scalar_t best_dist[THREADS];
  __shared__ int64_t best_dist_idx[THREADS];

  scalar_t best = 1e38;
  int64_t best_idx = 0;
  for (int64_t n_y = y_start_idx + thread_idx; n_y < y_end_idx;
       n_y += THREADS) {
    scalar_t dist = 0;
    for (int64_t d = 0; d < dim; d++) {
      dist += (x[n_x * dim + d] - y[n_y * dim + d]) *
              (x[n_x * dim + d] - y[n_y * dim + d]);
    }

    if (dist < best) {
      best = dist;
      best_idx = n_y;
    }
  }

  best_dist[thread_idx] = best;
  best_dist_idx[thread_idx] = best_idx;

  for (int64_t u = 0; (1 << u) < THREADS; u++) {
    __syncthreads();
    if (thread_idx < (THREADS >> (u + 1))) {
      int64_t idx_1 = (thread_idx * 2) << u;
      int64_t idx_2 = (thread_idx * 2 + 1) << u;
      if (best_dist[idx_1] > best_dist[idx_2]) {
        best_dist[idx_1] = best_dist[idx_2];
        best_dist_idx[idx_1] = best_dist_idx[idx_2];
      }
    }
  }

  __syncthreads();
  if (thread_idx == 0) {
    out[n_x] = best_dist_idx[0];
  }
}

torch::Tensor nearest_cuda(torch::Tensor x, torch::Tensor y,
                           torch::Tensor ptr_x, torch::Tensor ptr_y) {
  CHECK_CUDA(x);
  CHECK_CUDA(y);
  CHECK_CUDA(ptr_x);
  CHECK_CUDA(ptr_y);
  cudaSetDevice(x.get_device());

  x = x.view({x.size(0), -1}).contiguous();
  y = y.view({y.size(0), -1}).contiguous();

  auto out = torch::empty({x.size(0)}, ptr_x.options());

  auto stream = at::cuda::getCurrentCUDAStream();
  auto scalar_type = x.scalar_type();
  AT_DISPATCH_FLOATING_TYPES_AND(at::ScalarType::Half, scalar_type, "_", [&] {
    nearest_kernel<scalar_t><<<x.size(0), THREADS, 0, stream>>>(
        x.data_ptr<scalar_t>(), y.data_ptr<scalar_t>(),
        ptr_x.data_ptr<int64_t>(), ptr_y.data_ptr<int64_t>(),
        out.data_ptr<int64_t>(), ptr_x.size(0) - 1, x.size(1));
  });

  return out;
}
