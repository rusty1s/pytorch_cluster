#include "radius_cuda.h"

#include <ATen/cuda/CUDAContext.h>

#include "utils.cuh"

#define THREADS 1024

template <typename scalar_t> struct Cosine {
  static inline __device__ scalar_t dot(const scalar_t *a, const scalar_t *b,
                                        int64_t size) {
    scalar_t result = 0;
    for (int64_t i = 0; i < size; i++) {
      result += a[i] * b[i];
    }
    return result;
  }

  static inline __device__ scalar_t norm(const scalar_t *a, int64_t size) {
    scalar_t result = 0;
    for (int64_t i = 0; i < size; i++) {
      result += a[i] * a[i];
    }
    return sqrt(result);
  }
};

template <typename scalar_t>
__global__ void knn_kernel(const scalar_t *x, const scalar_t *y,
                           const int64_t *ptr_x, const int64_t *ptr_y,
                           scalar_t *dist, int64_t *row, int64_t *col,
                           int64_t K, int64_t dim, bool cosine) {

  const int64_t batch_idx = blockIdx.x;

  const int64_t x_start_idx = ptr_x[batch_idx];
  const int64_t x_end_idx = ptr_x[batch_idx + 1];

  const int64_t y_start_idx = ptr_y[batch_idx];
  const int64_t y_end_idx = ptr_y[batch_idx + 1];

  for (int64_t n_y = y_start_idx + threadIdx.x; n_y < y_end_idx;
       n_y += THREADS) {

    for (int64_t k = 0; k < K; k++) {
      row[n_y * K + k] = n_y;
    }

    for (int64_t n_x = x_start_idx; n_x < x_end_idx; n_x++) {

      scalar_t tmp_dist = 0;
      if (cosine) {
        tmp_dist =
            Cosine<scalar_t>::norm(x, dim) * Cosine<scalar_t>::norm(y, dim) -
            Cosine<scalar_t>::dot(x, y, dim);
      } else {
        for (int64_t d = 0; d < dim; d++) {
          tmp_dist += (x[n_x * dim + d] - y[n_y * dim + d]) *
                      (x[n_x * dim + d] - y[n_y * dim + d]);
        }
      }

      for (int64_t k_idx_1 = 0; k_idx_1 < K; k_idx_1++) {
        if (dist[n_y * K + k_idx_1] > tmp_dist) {
          for (ptrdiff_t k_idx_2 = K - 1; k_idx_2 > k_idx_1; k_idx_2--) {
            dist[n_y * K + k_idx_2] = dist[n_y * K + k_idx_2 - 1];
            col[n_y * K + k_idx_2] = col[n_y * K + k_idx_2 - 1];
          }
          dist[n_y * K + k_idx_1] = tmp_dist;
          col[n_y * K + k_idx_1] = n_x;
          break;
        }
      }
    }
  }
}

torch::Tensor knn_cuda(torch::Tensor x, torch::Tensor y,
                       torch::optional<torch::Tensor> ptr_x,
                       torch::optional<torch::Tensor> ptr_y, int64_t k,
                       bool cosine) {

  CHECK_CUDA(x);
  CHECK_INPUT(x.dim() == 2);
  CHECK_CUDA(y);
  CHECK_INPUT(y.dim() == 2);
  cudaSetDevice(x.get_device());

  if (ptr_x.has_value()) {
    CHECK_CUDA(ptr_x.value());
    CHECK_INPUT(ptr_x.value().dim() == 1);
  } else {
    ptr_x = torch::arange(0, x.size(0) + 1, x.size(0),
                          x.options().dtype(torch::kLong));
  }
  if (ptr_y.has_value()) {
    CHECK_CUDA(ptr_y.value());
    CHECK_INPUT(ptr_y.value().dim() == 1);
  } else {
    ptr_y = torch::arange(0, y.size(0) + 1, y.size(0),
                          y.options().dtype(torch::kLong));
  }
  CHECK_INPUT(ptr_x.value().numel() == ptr_y.value().numel());

  auto dist = torch::full(y.size(0) * k, 1e38, y.options());
  auto row = torch::empty(y.size(0) * k, ptr_y.value().options());
  auto col = torch::full(y.size(0) * k, -1, ptr_y.value().options());

  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "knn_kernel", [&] {
    knn_kernel<scalar_t><<<ptr_x.value().size(0) - 1, THREADS, 0, stream>>>(
        x.data_ptr<scalar_t>(), y.data_ptr<scalar_t>(),
        ptr_x.value().data_ptr<int64_t>(), ptr_y.value().data_ptr<int64_t>(),
        dist.data_ptr<scalar_t>(), row.data_ptr<int64_t>(),
        col.data_ptr<int64_t>(), k, x.size(1), cosine);
  });

  auto mask = col != -1;
  return torch::stack({row.masked_select(mask), col.masked_select(mask)}, 0);
}
