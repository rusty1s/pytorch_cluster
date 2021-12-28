#include "fps_cuda.h"

#include <ATen/cuda/CUDAContext.h>

#include "utils.cuh"

#define THREADS 256

template <typename scalar_t>
__global__ void fps_kernel(const scalar_t *src, const int64_t *ptr,
                           const int64_t *out_ptr, const int64_t *start,
                           scalar_t *dist, int64_t *out, int64_t dim) {

  const int64_t thread_idx = threadIdx.x;
  const int64_t batch_idx = blockIdx.x;

  const int64_t start_idx = ptr[batch_idx];
  const int64_t end_idx = ptr[batch_idx + 1];

  __shared__ scalar_t best_dist[THREADS];
  __shared__ int64_t best_dist_idx[THREADS];

  if (thread_idx == 0) {
    out[out_ptr[batch_idx]] = start_idx + start[batch_idx];
  }

  for (int64_t m = out_ptr[batch_idx] + 1; m < out_ptr[batch_idx + 1]; m++) {
    __syncthreads();
    int64_t old = out[m - 1];

    scalar_t best = (scalar_t)-1.;
    int64_t best_idx = 0;

    for (int64_t n = start_idx + thread_idx; n < end_idx; n += THREADS) {
      scalar_t tmp, dd = (scalar_t)0.;
      for (int64_t d = 0; d < dim; d++) {
        tmp = src[dim * old + d] - src[dim * n + d];
        dd += tmp * tmp;
      }
      dd = min(dist[n], dd);
      dist[n] = dd;
      if (dd > best) {
        best = dd;
        best_idx = n;
      }
    }

    best_dist[thread_idx] = best;
    best_dist_idx[thread_idx] = best_idx;

    for (int64_t i = 1; i < THREADS; i *= 2) {
      __syncthreads();
      if ((thread_idx + i) < THREADS &&
          best_dist[thread_idx] < best_dist[thread_idx + i]) {
        best_dist[thread_idx] = best_dist[thread_idx + i];
        best_dist_idx[thread_idx] = best_dist_idx[thread_idx + i];
      }
    }

    __syncthreads();
    if (thread_idx == 0) {
      out[m] = best_dist_idx[0];
    }
  }
}

torch::Tensor fps_cuda(torch::Tensor src, torch::Tensor ptr,
                       torch::Tensor ratio, bool random_start) {

  CHECK_CUDA(src);
  CHECK_CUDA(ptr);
  CHECK_CUDA(ratio);
  CHECK_INPUT(ptr.dim() == 1);
  cudaSetDevice(src.get_device());

  src = src.view({src.size(0), -1}).contiguous();
  ptr = ptr.contiguous();
  auto batch_size = ptr.numel() - 1;

  auto deg = ptr.narrow(0, 1, batch_size) - ptr.narrow(0, 0, batch_size);
  auto out_ptr = deg.toType(ratio.scalar_type()) * ratio;
  out_ptr = out_ptr.ceil().toType(torch::kLong).cumsum(0);
  out_ptr = torch::cat({torch::zeros(1, ptr.options()), out_ptr}, 0);

  torch::Tensor start;
  if (random_start) {
    start = torch::rand(batch_size, src.options());
    start = (start * deg.toType(ratio.scalar_type())).toType(torch::kLong);
  } else {
    start = torch::zeros(batch_size, ptr.options());
  }

  auto dist = torch::full(src.size(0), 5e4, src.options());

  auto out_size = (int64_t *)malloc(sizeof(int64_t));
  cudaMemcpy(out_size, out_ptr[-1].data_ptr<int64_t>(), sizeof(int64_t),
             cudaMemcpyDeviceToHost);
  auto out = torch::empty(out_size[0], out_ptr.options());

  auto stream = at::cuda::getCurrentCUDAStream();
  auto scalar_type = src.scalar_type();
  AT_DISPATCH_FLOATING_TYPES_AND(at::ScalarType::Half, scalar_type, "_", [&] {
    fps_kernel<scalar_t><<<batch_size, THREADS, 0, stream>>>(
        src.data_ptr<scalar_t>(), ptr.data_ptr<int64_t>(),
        out_ptr.data_ptr<int64_t>(), start.data_ptr<int64_t>(),
        dist.data_ptr<scalar_t>(), out.data_ptr<int64_t>(), src.size(1));
  });

  return out;
}
