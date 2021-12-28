#include "grid_cuda.h"

#include <ATen/cuda/CUDAContext.h>

#include "utils.cuh"

#define THREADS 1024
#define BLOCKS(N) (N + THREADS - 1) / THREADS

template <typename scalar_t>
__global__ void grid_kernel(const scalar_t *pos, const scalar_t *size,
                            const scalar_t *start, const scalar_t *end,
                            int64_t *out, int64_t D, int64_t numel) {
  const int64_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (thread_idx < numel) {
    int64_t c = 0, k = 1;
    for (int64_t d = 0; d < D; d++) {
      scalar_t p = pos[thread_idx * D + d] - start[d];
      c += (int64_t)(p / size[d]) * k;
      k *= (int64_t)((end[d] - start[d]) / size[d]) + 1;
    }
    out[thread_idx] = c;
  }
}

torch::Tensor grid_cuda(torch::Tensor pos, torch::Tensor size,
                        torch::optional<torch::Tensor> optional_start,
                        torch::optional<torch::Tensor> optional_end) {
  CHECK_CUDA(pos);
  CHECK_CUDA(size);
  cudaSetDevice(pos.get_device());

  if (optional_start.has_value())
    CHECK_CUDA(optional_start.value());
  if (optional_start.has_value())
    CHECK_CUDA(optional_start.value());

  pos = pos.view({pos.size(0), -1}).contiguous();
  size = size.contiguous();

  CHECK_INPUT(size.numel() == pos.size(1));

  if (!optional_start.has_value())
    optional_start = std::get<0>(pos.min(0));
  else {
    optional_start = optional_start.value().contiguous();
    CHECK_INPUT(optional_start.value().numel() == pos.size(1));
  }

  if (!optional_end.has_value())
    optional_end = std::get<0>(pos.max(0));
  else {
    optional_start = optional_start.value().contiguous();
    CHECK_INPUT(optional_start.value().numel() == pos.size(1));
  }

  auto start = optional_start.value();
  auto end = optional_end.value();

  auto out = torch::empty(pos.size(0), pos.options().dtype(torch::kLong));

  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, pos.scalar_type(), "_", [&] {
    grid_kernel<scalar_t><<<BLOCKS(out.numel()), THREADS, 0, stream>>>(
        pos.data_ptr<scalar_t>(), size.data_ptr<scalar_t>(),
        start.data_ptr<scalar_t>(), end.data_ptr<scalar_t>(),
        out.data_ptr<int64_t>(), pos.size(1), out.numel());
  });

  return out;
}
