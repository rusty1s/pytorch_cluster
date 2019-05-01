#include <ATen/ATen.h>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/detail/TensorInfo.cuh>

#define THREADS 1024
#define BLOCKS(N) (N + THREADS - 1) / THREADS

template <typename scalar_t>
__global__ void grid_kernel(int64_t *cluster,
                            at::cuda::detail::TensorInfo<scalar_t, int64_t> pos,
                            scalar_t *__restrict__ size,
                            scalar_t *__restrict__ start,
                            scalar_t *__restrict__ end, size_t numel) {
  const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t stride = blockDim.x * gridDim.x;
  for (ptrdiff_t i = index; i < numel; i += stride) {
    int64_t c = 0, k = 1;
    for (ptrdiff_t d = 0; d < pos.sizes[1]; d++) {
      scalar_t p = pos.data[i * pos.strides[0] + d * pos.strides[1]] - start[d];
      c += (int64_t)(p / size[d]) * k;
      k *= (int64_t)((end[d] - start[d]) / size[d]) + 1;
    }
    cluster[i] = c;
  }
}

at::Tensor grid_cuda(at::Tensor pos, at::Tensor size, at::Tensor start,
                     at::Tensor end) {
  cudaSetDevice(pos.get_device());
  auto cluster = at::empty(pos.size(0), pos.options().dtype(at::kLong));

  AT_DISPATCH_ALL_TYPES(pos.scalar_type(), "grid_kernel", [&] {
    grid_kernel<scalar_t><<<BLOCKS(cluster.numel()), THREADS>>>(
        cluster.data<int64_t>(),
        at::cuda::detail::getTensorInfo<scalar_t, int64_t>(pos),
        size.data<scalar_t>(), start.data<scalar_t>(), end.data<scalar_t>(),
        cluster.numel());
  });

  return cluster;
}
