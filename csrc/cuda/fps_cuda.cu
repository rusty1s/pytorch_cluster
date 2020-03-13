#include "fps_cuda.h"

#include <ATen/cuda/CUDAContext.h>

#include "atomics.cuh"
#include "utils.cuh"

#define THREADS 1024

template <typename scalar_t> struct Dist {
  static inline __device__ void compute(int64_t idx, int64_t start_idx,
                                        int64_t end_idx, int64_t old,
                                        scalar_t *best, int64_t *best_idx,
                                        const scalar_t *src, scalar_t *dist,
                                        scalar_t *tmp_dist, int64_t dim) {

    for (int64_t n = start_idx + idx; n < end_idx; n += THREADS) {
      tmp_dist[n] = 0;
    }

    __syncthreads();
    for (int64_t i = start_idx * dim + idx; i < end_idx * dim; i += THREADS) {
      scalar_t d = src[(old * dim) + (i % dim)] - src[i];
      atomAdd(&tmp_dist[i / dim], d * d);
    }

    __syncthreads();
    for (int64_t n = start_idx + idx; n < end_idx; n += THREADS) {
      dist[n] = min(dist[n], tmp_dist[n]);
      if (dist[n] > *best) {
        *best = dist[n];
        *best_idx = n;
      }
    }
  }
};

template <typename scalar_t>
__global__ void fps_kernel(const scalar_t *src, const int64_t *ptr,
                           const int64_t *out_ptr, const int64_t *start,
                           scalar_t *dist, scalar_t *tmp_dist, int64_t *out,
                           int64_t dim) {

  const int64_t batch_idx = blockIdx.x;
  const int64_t thread_idx = threadIdx.x;

  const int64_t start_idx = ptr[batch_idx];
  const int64_t end_idx = ptr[batch_idx + 1];

  __shared__ scalar_t best_dist[THREADS];
  __shared__ int64_t best_dist_idx[THREADS];

  if (threadIdx.x == 0) {
    out[out_ptr[batch_idx]] = start_idx + start[batch_idx];
  }

  for (int64_t m = out_ptr[batch_idx] + 1; m < out_ptr[batch_idx + 1]; m++) {
    scalar_t best = -1;
    int64_t best_idx = 0;

    __syncthreads();
    Dist<scalar_t>::compute(thread_idx, start_idx, end_idx, out[m - 1], &best,
                            &best_idx, src, dist, tmp_dist, dim);

    best_dist[thread_idx] = best;
    best_dist_idx[thread_idx] = best_idx;

    for (int64_t u = 0; (1 << u) < THREADS; u++) {
      __syncthreads();
      if (thread_idx < (THREADS >> (u + 1))) {
        int64_t idx1 = (thread_idx * 2) << u;
        int64_t idx2 = (thread_idx * 2 + 1) << u;
        if (best_dist[idx1] < best_dist[idx2]) {
          best_dist[idx1] = best_dist[idx2];
          best_dist_idx[idx1] = best_dist_idx[idx2];
        }
      }
    }

    __syncthreads();
    if (thread_idx == 0) {
      out[m] = best_dist_idx[0];
    }
  }
}

torch::Tensor fps_cuda(torch::Tensor src, torch::Tensor ptr, double ratio,
                       bool random_start) {

  CHECK_CUDA(src);
  CHECK_CUDA(ptr);
  CHECK_INPUT(ptr.dim() == 1);
  AT_ASSERTM(ratio > 0 and ratio < 1, "Invalid input");
  cudaSetDevice(src.get_device());

  src = src.view({src.size(0), -1}).contiguous();
  ptr = ptr.contiguous();
  auto batch_size = ptr.size(0) - 1;

  auto deg = ptr.narrow(0, 1, batch_size) - ptr.narrow(0, 0, batch_size);
  auto out_ptr = deg.toType(torch::kFloat) * (float)ratio;
  out_ptr = out_ptr.ceil().toType(torch::kLong).cumsum(0);
  out_ptr = torch::cat({torch::zeros(1, ptr.options()), out_ptr}, 0);

  torch::Tensor start;
  if (random_start) {
    start = torch::rand(batch_size, src.options());
    start = (start * deg.toType(torch::kFloat)).toType(torch::kLong);
  } else {
    start = torch::zeros(batch_size, ptr.options());
  }

  auto dist = torch::full(src.size(0), 1e38, src.options());
  auto tmp_dist = torch::empty(src.size(0), src.options());

  auto out_size = (int64_t *)malloc(sizeof(int64_t));
  cudaMemcpy(out_size, out_ptr[-1].data_ptr<int64_t>(), sizeof(int64_t),
             cudaMemcpyDeviceToHost);
  auto out = torch::empty(out_size[0], out_ptr.options());

  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES(src.scalar_type(), "fps_kernel", [&] {
    fps_kernel<scalar_t><<<batch_size, THREADS, 0, stream>>>(
        src.data_ptr<scalar_t>(), ptr.data_ptr<int64_t>(),
        out_ptr.data_ptr<int64_t>(), start.data_ptr<int64_t>(),
        dist.data_ptr<scalar_t>(), tmp_dist.data_ptr<scalar_t>(),
        out.data_ptr<int64_t>(), src.size(1));
  });

  return out;
}
