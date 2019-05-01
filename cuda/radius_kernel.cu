#include <ATen/ATen.h>

#include "utils.cuh"

#define THREADS 1024

template <typename scalar_t>
__global__ void
radius_kernel(const scalar_t *__restrict__ x, const scalar_t *__restrict__ y,
              const int64_t *__restrict__ batch_x,
              const int64_t *__restrict__ batch_y, int64_t *__restrict__ row,
              int64_t *__restrict__ col, scalar_t radius,
              size_t max_num_neighbors, size_t dim) {

  const ptrdiff_t batch_idx = blockIdx.x;
  const ptrdiff_t idx = threadIdx.x;

  const ptrdiff_t start_idx_x = batch_x[batch_idx];
  const ptrdiff_t end_idx_x = batch_x[batch_idx + 1];

  const ptrdiff_t start_idx_y = batch_y[batch_idx];
  const ptrdiff_t end_idx_y = batch_y[batch_idx + 1];

  for (ptrdiff_t n_y = start_idx_y + idx; n_y < end_idx_y; n_y += THREADS) {
    size_t count = 0;
    for (ptrdiff_t n_x = start_idx_x; n_x < end_idx_x; n_x++) {

      scalar_t dist = 0;
      for (ptrdiff_t d = 0; d < dim; d++) {
        dist += (x[n_x * dim + d] - y[n_y * dim + d]) *
                (x[n_x * dim + d] - y[n_y * dim + d]);
      }
      dist = sqrt(dist);

      if (dist <= radius) {
        row[n_y * max_num_neighbors + count] = n_y;
        col[n_y * max_num_neighbors + count] = n_x;
        count++;
      }

      if (count >= max_num_neighbors) {
        break;
      }
    }
  }
}

at::Tensor radius_cuda(at::Tensor x, at::Tensor y, float radius,
                       at::Tensor batch_x, at::Tensor batch_y,
                       size_t max_num_neighbors) {
  cudaSetDevice(x.get_device());
  auto batch_sizes = (int64_t *)malloc(sizeof(int64_t));
  cudaMemcpy(batch_sizes, batch_x[-1].data<int64_t>(), sizeof(int64_t),
             cudaMemcpyDeviceToHost);
  auto batch_size = batch_sizes[0] + 1;

  batch_x = degree(batch_x, batch_size);
  batch_x = at::cat({at::zeros(1, batch_x.options()), batch_x.cumsum(0)}, 0);
  batch_y = degree(batch_y, batch_size);
  batch_y = at::cat({at::zeros(1, batch_y.options()), batch_y.cumsum(0)}, 0);

  auto row = at::full(y.size(0) * max_num_neighbors, -1, batch_y.options());
  auto col = at::full(y.size(0) * max_num_neighbors, -1, batch_y.options());

  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "radius_kernel", [&] {
    radius_kernel<scalar_t><<<batch_size, THREADS>>>(
        x.data<scalar_t>(), y.data<scalar_t>(), batch_x.data<int64_t>(),
        batch_y.data<int64_t>(), row.data<int64_t>(), col.data<int64_t>(),
        radius, max_num_neighbors, x.size(1));
  });

  auto mask = row != -1;
  return at::stack({row.masked_select(mask), col.masked_select(mask)}, 0);
}
