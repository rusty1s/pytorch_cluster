#include <ATen/ATen.h>

#include "utils.cuh"

#define THREADS 1024

template <typename scalar_t> struct Cosine {
  static inline __device__ scalar_t dot(const scalar_t *a, const scalar_t *b,
                                        size_t size) {
    scalar_t result = 0;
    for (ptrdiff_t i = 0; i < size; i++) {
      result += a[i] * b[i];
    }
    return result;
  }

  static inline __device__ scalar_t norm(const scalar_t *a, size_t size) {
    scalar_t result = 0;
    for (ptrdiff_t i = 0; i < size; i++) {
      result += a[i] * a[i];
    }
    return sqrt(result);
  }
};

template <typename scalar_t>
__global__ void
knn_kernel(const scalar_t *__restrict__ x, const scalar_t *__restrict__ y,
           const int64_t *__restrict__ batch_x,
           const int64_t *__restrict__ batch_y, scalar_t *__restrict__ dist,
           int64_t *__restrict__ row, int64_t *__restrict__ col, size_t k,
           size_t dim, bool cosine) {

  const ptrdiff_t batch_idx = blockIdx.x;
  const ptrdiff_t idx = threadIdx.x;

  const ptrdiff_t start_idx_x = batch_x[batch_idx];
  const ptrdiff_t end_idx_x = batch_x[batch_idx + 1];

  const ptrdiff_t start_idx_y = batch_y[batch_idx];
  const ptrdiff_t end_idx_y = batch_y[batch_idx + 1];

  for (ptrdiff_t n_y = start_idx_y + idx; n_y < end_idx_y; n_y += THREADS) {

    for (ptrdiff_t k_idx = 0; k_idx < k; k_idx++) {
      row[n_y * k + k_idx] = n_y;
    }

    for (ptrdiff_t n_x = start_idx_x; n_x < end_idx_x; n_x++) {

      scalar_t tmp_dist = 0;
      if (cosine) {
        tmp_dist =
            Cosine<scalar_t>::norm(x, dim) * Cosine<scalar_t>::norm(y, dim) -
            Cosine<scalar_t>::dot(x, y, dim);
      } else {
        for (ptrdiff_t d = 0; d < dim; d++) {
          tmp_dist += (x[n_x * dim + d] - y[n_y * dim + d]) *
                      (x[n_x * dim + d] - y[n_y * dim + d]);
        }
      }

      for (ptrdiff_t k_idx_1 = 0; k_idx_1 < k; k_idx_1++) {
        if (dist[n_y * k + k_idx_1] > tmp_dist) {
          for (ptrdiff_t k_idx_2 = k - 1; k_idx_2 > k_idx_1; k_idx_2--) {
            dist[n_y * k + k_idx_2] = dist[n_y * k + k_idx_2 - 1];
            col[n_y * k + k_idx_2] = col[n_y * k + k_idx_2 - 1];
          }
          dist[n_y * k + k_idx_1] = tmp_dist;
          col[n_y * k + k_idx_1] = n_x;
          break;
        }
      }
    }
  }
}

at::Tensor knn_cuda(at::Tensor x, at::Tensor y, size_t k, at::Tensor batch_x,
                    at::Tensor batch_y, bool cosine) {
  cudaSetDevice(x.get_device());
  auto batch_sizes = (int64_t *)malloc(sizeof(int64_t));
  cudaMemcpy(batch_sizes, batch_x[-1].data<int64_t>(), sizeof(int64_t),
             cudaMemcpyDeviceToHost);
  auto batch_size = batch_sizes[0] + 1;

  batch_x = degree(batch_x, batch_size);
  batch_x = at::cat({at::zeros(1, batch_x.options()), batch_x.cumsum(0)}, 0);
  batch_y = degree(batch_y, batch_size);
  batch_y = at::cat({at::zeros(1, batch_y.options()), batch_y.cumsum(0)}, 0);

  auto dist = at::full(y.size(0) * k, 1e38, y.options());
  auto row = at::empty(y.size(0) * k, batch_y.options());
  auto col = at::full(y.size(0) * k, -1, batch_y.options());

  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "knn_kernel", [&] {
    knn_kernel<scalar_t><<<batch_size, THREADS>>>(
        x.data<scalar_t>(), y.data<scalar_t>(), batch_x.data<int64_t>(),
        batch_y.data<int64_t>(), dist.data<scalar_t>(), row.data<int64_t>(),
        col.data<int64_t>(), k, x.size(1), cosine);
  });

  auto mask = col != -1;
  return at::stack({row.masked_select(mask), col.masked_select(mask)}, 0);
}
