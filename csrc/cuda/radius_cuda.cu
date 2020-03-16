#include "radius_cuda.h"

#include <ATen/cuda/CUDAContext.h>

#include "utils.cuh"

#define THREADS 1024

template <typename scalar_t>
__global__ void radius_kernel(const scalar_t *x, const scalar_t *y,
                              const int64_t *ptr_x, const int64_t *ptr_y,
                              int64_t *row, int64_t *col, scalar_t radius,
                              int64_t max_num_neighbors, int64_t dim) {

  const int64_t batch_idx = blockIdx.x;

  const int64_t x_start_idx = ptr_x[batch_idx];
  const int64_t x_end_idx = ptr_x[batch_idx + 1];

  const int64_t y_start_idx = ptr_y[batch_idx];
  const int64_t y_end_idx = ptr_y[batch_idx + 1];

  for (int64_t n_y = y_start_idx + threadIdx.x; n_y < y_end_idx;
       n_y += THREADS) {
    int64_t count = 0;
    for (int64_t n_x = x_start_idx; n_x < x_end_idx; n_x++) {
      scalar_t dist = 0;
      for (int64_t d = 0; d < dim; d++) {
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

torch::Tensor radius_cuda(torch::Tensor x, torch::Tensor y, torch::Tensor ptr_x,
                          torch::Tensor ptr_y, double r,
                          int64_t max_num_neighbors) {
  CHECK_CUDA(x);
  CHECK_CUDA(y);
  CHECK_CUDA(ptr_x);
  CHECK_CUDA(ptr_y);
  cudaSetDevice(x.get_device());

  x = x.view({x.size(0), -1}).contiguous();
  y = y.view({y.size(0), -1}).contiguous();

  auto row = torch::full(y.size(0) * max_num_neighbors, -1, ptr_y.options());
  auto col = torch::full(y.size(0) * max_num_neighbors, -1, ptr_y.options());

  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "radius_kernel", [&] {
    radius_kernel<scalar_t><<<ptr_x.size(0) - 1, THREADS, 0, stream>>>(
        x.data_ptr<scalar_t>(), y.data_ptr<scalar_t>(),
        ptr_x.data_ptr<int64_t>(), ptr_y.data_ptr<int64_t>(),
        row.data_ptr<int64_t>(), col.data_ptr<int64_t>(), r, max_num_neighbors,
        x.size(1));
  });

  auto mask = row != -1;
  return torch::stack({row.masked_select(mask), col.masked_select(mask)}, 0);
}
