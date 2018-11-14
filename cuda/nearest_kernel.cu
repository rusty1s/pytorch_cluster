#include <ATen/ATen.h>

#include "utils.cuh"

#define THREADS 1024

template <typename scalar_t>
__global__ void
nearest_kernel(scalar_t *__restrict__ x, scalar_t *__restrict__ y,
               int64_t *__restrict__ batch_x, int64_t *__restrict__ batch_y,
               scalar_t *__restrict__ out, int64_t *__restrict__ out_idx,
               size_t dim) {

  const ptrdiff_t n_x = blockIdx.x;
  const ptrdiff_t batch_idx = batch_x[n_x];
  const ptrdiff_t idx = threadIdx.x;

  const ptrdiff_t start_idx = batch_y[batch_idx];
  const ptrdiff_t end_idx = batch_y[batch_idx + 1];

  __shared__ scalar_t best_dist[THREADS];
  __shared__ int64_t best_dist_idx[THREADS];

  scalar_t best = 1e38;
  ptrdiff_t best_idx = 0;
  for (ptrdiff_t n_y = start_idx + idx; n_y < end_idx; n_y += THREADS) {

    scalar_t dist = 0;
    for (ptrdiff_t d = 0; d < dim; d++) {
      dist += (x[n_x * dim + d] - y[n_y * dim + d]) *
              (x[n_x * dim + d] - y[n_y * dim + d]);
    }
    dist = sqrt(dist);

    if (dist < best) {
      best = dist;
      best_idx = n_y;
    }
  }

  best_dist[idx] = best;
  best_dist_idx[idx] = best_idx;

  for (int64_t u = 0; (1 << u) < THREADS; u++) {
    __syncthreads();
    if (idx < (THREADS >> (u + 1))) {
      int64_t idx_1 = (idx * 2) << u;
      int64_t idx_2 = (idx * 2 + 1) << u;
      if (best_dist[idx_1] > best_dist[idx_2]) {
        best_dist[idx_1] = best_dist[idx_2];
        best_dist_idx[idx_1] = best_dist_idx[idx_2];
      }
    }
  }

  __syncthreads();
  if (idx == 0) {
    out[n_x] = best_dist[0];
    out_idx[n_x] = best_dist_idx[0];
  }
}

std::tuple<at::Tensor, at::Tensor> nearest_cuda(at::Tensor x, at::Tensor y,
                                                at::Tensor batch_x,
                                                at::Tensor batch_y) {
  auto batch_sizes = (int64_t *)malloc(sizeof(int64_t));
  cudaMemcpy(batch_sizes, batch_x[-1].data<int64_t>(), sizeof(int64_t),
             cudaMemcpyDeviceToHost);
  auto batch_size = batch_sizes[0] + 1;

  batch_y = degree(batch_y, batch_size);
  batch_y = at::cat({at::zeros(1, batch_y.options()), batch_y.cumsum(0)}, 0);

  auto out = at::empty(x.size(0), x.options());
  auto out_idx = at::empty_like(batch_x);

  AT_DISPATCH_FLOATING_TYPES(x.type(), "fps_kernel", [&] {
    nearest_kernel<scalar_t><<<x.size(0), THREADS>>>(
        x.data<scalar_t>(), y.data<scalar_t>(), batch_x.data<int64_t>(),
        batch_y.data<int64_t>(), out.data<scalar_t>(), out_idx.data<int64_t>(),
        x.size(1));
  });

  return std::make_tuple(out, out_idx);
}
