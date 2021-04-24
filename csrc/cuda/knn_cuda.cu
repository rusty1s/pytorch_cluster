#include "radius_cuda.h"

#include <ATen/cuda/CUDAContext.h>

#include "utils.cuh"

#define THREADS 1024

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
__global__ void knn_kernel(const scalar_t *x, const scalar_t *y,
                           const int64_t *ptr_x, const int64_t *batch_y,
                           scalar_t *dists, int64_t *row, int64_t *col,
                           int64_t K, int64_t n_points, int64_t dim, bool cosine) {

    const int64_t i = threadIdx.x + blockIdx.y * blockDim.x + blockIdx.x * gridDim.y * blockDim.x;
    if (i >= n_points)
        return;

    const int64_t b = batch_y[i];
    const int64_t start_x = ptr_x[b];
    const int64_t end_x = ptr_x[b + 1];
    for (int64_t k = 0; k < K; k++)
        row[i * K + k] = i;

    for (int64_t j = start_x; j < end_x; j++) {
        scalar_t dist = 0.;
        for (int d = 0; d < dim; d++)
            dist += (x[j * dim + d] - y[i * dim + d]) * (x[j * dim + d] - y[i * dim + d]);

        for (int k = 0; k < K; k++) {
            if (dist < dists[i * K + k]) {
                for (int idx = K - 1; idx > k; idx--) {
                    dists[i * K + idx] = dists[i * K + idx - 1];
                    col[i * K + idx] = col[i * K + idx - 1];
                }

                dists[i * K + k] = dist;
                col[i * K + k] = j;
                break;
            }
        }
    }
}

torch::Tensor knn_cuda(torch::Tensor x, torch::Tensor y,
                       torch::optional<torch::Tensor> ptr_x,
                       torch::optional<torch::Tensor> batch_y, int64_t k,
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

    if (batch_y.has_value()) {
        CHECK_CUDA(batch_y.value());
        CHECK_INPUT(batch_y.value().dim() == 1);
    } else {
        batch_y = torch::zeros({x.size(0)}, x.options().dtype(torch::kLong));
    }

//    CHECK_INPUT(ptr_x.value().numel() == batch_y.value().numel());

    auto dist = torch::full(y.size(0) * k, 1e38, y.options());
    auto row = torch::empty(y.size(0) * k, batch_y.value().options());
    auto col = torch::full(y.size(0) * k, -1, batch_y.value().options());
    auto bs = torch::max(batch_y.value()).item<int>() + 1;
    dim3 BLOCKS(bs, (int) ceil(((float) y.size(0) / THREADS) / bs));

    auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "knn_kernel", [&] {
        knn_kernel<scalar_t><<<BLOCKS, THREADS, 0, stream>>>(
            x.data_ptr<scalar_t>(), y.data_ptr<scalar_t>(),
                    ptr_x.value().data_ptr<int64_t>(), batch_y.value().data_ptr<int64_t>(),
                    dist.data_ptr<scalar_t>(), row.data_ptr<int64_t>(),
                    col.data_ptr<int64_t>(), k, y.size(0), x.size(1), cosine);
    });

    auto mask = col != -1;
    return torch::stack({row.masked_select(mask), col.masked_select(mask)}, 0);
}
