#include "radius_cuda.h"

#include <ATen/cuda/CUDAContext.h>

#include "utils.cuh"

#define THREADS 256

template <typename scalar_t> struct Cosine {
  static inline __device__ scalar_t dot(const scalar_t *a, const scalar_t *b,
                                        int64_t n_a, int64_t n_b,
                                        int64_t size) {
    scalar_t result = 0;
    for (int64_t i = 0; i < size; i++) {
      result += a[n_a * size + i] * b[n_b * size + i];
    }
    return result;
  }

  static inline __device__ scalar_t norm(const scalar_t *a, int64_t n_a,
                                         int64_t size) {
    scalar_t result = 0;
    for (int64_t i = 0; i < size; i++) {
      result += a[n_a * size + i] * a[n_a * size + i];
    }
    return sqrt(result);
  }
};

template <typename scalar_t>
__global__ void
knn_kernel(const scalar_t *__restrict__ x, const scalar_t *__restrict__ y,
           const int64_t *__restrict__ ptr_x, const int64_t *__restrict__ ptr_y,
           int64_t *__restrict__ row, int64_t *__restrict__ col,
           const int64_t k, const int64_t n, const int64_t m, const int64_t dim,
           const int64_t num_examples, const bool cosine) {

  const int64_t n_y = blockIdx.x * blockDim.x + threadIdx.x;
  if (n_y >= m)
    return;

  const int64_t example_idx = get_example_idx(n_y, ptr_y, num_examples);

  scalar_t best_dist[100];
  int64_t best_idx[100];

  for (int e = 0; e < k; e++) {
    best_dist[e] = 5e4;
    best_idx[e] = -1;
  }

  for (int64_t n_x = ptr_x[example_idx]; n_x < ptr_x[example_idx + 1]; n_x++) {
    scalar_t tmp_dist = 0;

    if (cosine) {
      tmp_dist = Cosine<scalar_t>::dot(x, y, n_x, n_y, dim) /
                 (Cosine<scalar_t>::norm(x, n_x, dim) *
                  Cosine<scalar_t>::norm(y, n_y, dim));
      tmp_dist = 1. - tmp_dist;
    } else {
      for (int64_t d = 0; d < dim; d++) {
        tmp_dist += (x[n_x * dim + d] - y[n_y * dim + d]) *
                    (x[n_x * dim + d] - y[n_y * dim + d]);
      }
    }

    for (int64_t e1 = 0; e1 < k; e1++) {
      if (best_dist[e1] > tmp_dist) {
        for (int64_t e2 = k - 1; e2 > e1; e2--) {
          best_dist[e2] = best_dist[e2 - 1];
          best_idx[e2] = best_idx[e2 - 1];
        }
        best_dist[e1] = tmp_dist;
        best_idx[e1] = n_x;
        break;
      }
    }
  }

  for (int64_t e = 0; e < k; e++) {
    row[n_y * k + e] = n_y;
    col[n_y * k + e] = best_idx[e];
  }
}

torch::Tensor knn_cuda(const torch::Tensor x, const torch::Tensor y,
                       torch::optional<torch::Tensor> ptr_x,
                       torch::optional<torch::Tensor> ptr_y, const int64_t k,
                       const bool cosine) {

  CHECK_CUDA(x);
  CHECK_CONTIGUOUS(x);
  CHECK_INPUT(x.dim() == 2);
  CHECK_CUDA(y);
  CHECK_CONTIGUOUS(y);
  CHECK_INPUT(y.dim() == 2);
  CHECK_INPUT(x.size(1) == y.size(1));
  AT_ASSERTM(k <= 100, "`k` needs to smaller than or equal to 100");

  if (ptr_x.has_value()) {
    CHECK_CUDA(ptr_x.value());
    CHECK_INPUT(ptr_x.value().dim() == 1);
  } else
    ptr_x = torch::arange(0, x.size(0) + 1, x.size(0),
                          x.options().dtype(torch::kLong));

  if (ptr_y.has_value()) {
    CHECK_CUDA(ptr_y.value());
    CHECK_INPUT(ptr_y.value().dim() == 1);
  } else
    ptr_y = torch::arange(0, y.size(0) + 1, y.size(0),
                          y.options().dtype(torch::kLong));

  CHECK_INPUT(ptr_x.value().numel() == ptr_y.value().numel());

  cudaSetDevice(x.get_device());

  auto row = torch::empty(y.size(0) * k, ptr_y.value().options());
  auto col = torch::full(y.size(0) * k, -1, ptr_y.value().options());

  dim3 BLOCKS((y.size(0) + THREADS - 1) / THREADS);

  auto stream = at::cuda::getCurrentCUDAStream();
  auto scalar_type = x.scalar_type();
  AT_DISPATCH_FLOATING_TYPES_AND(at::ScalarType::Half, scalar_type, "_", [&] {
    knn_kernel<scalar_t><<<BLOCKS, THREADS, 0, stream>>>(
        x.data_ptr<scalar_t>(), y.data_ptr<scalar_t>(),
        ptr_x.value().data_ptr<int64_t>(), ptr_y.value().data_ptr<int64_t>(),
        row.data_ptr<int64_t>(), col.data_ptr<int64_t>(), k, x.size(0),
        y.size(0), x.size(1), ptr_x.value().numel() - 1, cosine);
  });

  auto mask = col != -1;
  return torch::stack({row.masked_select(mask), col.masked_select(mask)}, 0);
}
