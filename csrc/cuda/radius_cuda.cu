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

torch::Tensor radius_cuda(torch::Tensor x, torch::Tensor y,
                          torch::optional<torch::Tensor> ptr_x,
                          torch::optional<torch::Tensor> ptr_y, double r,
                          int64_t max_num_neighbors) {
  CHECK_CUDA(x);
  CHECK_INPUT(x.dim() == 2);
  CHECK_CUDA(y);
  CHECK_INPUT(y.dim() == 2);
  cudaSetDevice(x.get_device());

  if (ptr_x.has_value()) {
    CHECK_CUDA(ptr_x.value());
    CHECK_INPUT(ptr_x.value().dim() == 1);
  } else {
    ptr_x = torch::tensor({0, x.size(0)}, x.options().dtype(torch::kLong));
  }
  if (ptr_y.has_value()) {
    CHECK_CUDA(ptr_y.value());
    CHECK_INPUT(ptr_y.value().dim() == 1);
  } else {
    ptr_y = torch::tensor({0, y.size(0)}, y.options().dtype(torch::kLong));
  }
  CHECK_INPUT(ptr_x.value().numel() == ptr_y.value().numel());

  auto row =
      torch::full(y.size(0) * max_num_neighbors, -1, ptr_y.value().options());
  auto col =
      torch::full(y.size(0) * max_num_neighbors, -1, ptr_y.value().options());

  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "radius_kernel", [&] {
    radius_kernel<scalar_t><<<ptr_x.size(0) - 1, THREADS, 0, stream>>>(
        x.data_ptr<scalar_t>(), y.data_ptr<scalar_t>(),
        ptr_x.value().data_ptr<int64_t>(), ptr_y.value().data_ptr<int64_t>(),
        row.data_ptr<int64_t>(), col.data_ptr<int64_t>(), r, max_num_neighbors,
        x.size(1));
  });

  auto mask = row != -1;
  return torch::stack({row.masked_select(mask), col.masked_select(mask)}, 0);
}
