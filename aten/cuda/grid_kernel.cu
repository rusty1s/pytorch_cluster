#include <ATen/ATen.h>
#include <ATen/cuda/detail/IndexUtils.cuh>

#define THREADS 1024
#define BLOCKS(N) (N + THREADS - 1) / THREADS

template <typename scalar_t>
__global__ void
grid_cuda_kernel(int64_t *cluster,
                 at::cuda::detail::TensorInfo<scalar_t, int> pos,
                 scalar_t *__restrict__ size, scalar_t *__restrict__ start,
                 scalar_t *__restrict__ end, size_t num_nodes) {
  const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t stride = blockDim.x * gridDim.x;
  for (ptrdiff_t i = index; i < num_nodes; i += stride) {
    int64_t c = 0, k = 1;
    scalar_t tmp;
    for (ptrdiff_t d = 0; d < pos.sizes[1]; d++) {
      tmp = (pos.data[i * pos.strides[0] + d * pos.strides[1]]) - start[d];
      c += (int64_t)(tmp / size[d]) * k;
      k += (int64_t)((end[d] - start[d]) / size[d]);
    }
    cluster[i] = c;
  }
}

at::Tensor grid_cuda(at::Tensor pos, at::Tensor size, at::Tensor start,
                     at::Tensor end) {
  auto num_nodes = pos.size(0);
  auto cluster = at::empty(pos.type().toScalarType(at::kLong), {num_nodes});

  AT_DISPATCH_ALL_TYPES(pos.type(), "grid_cuda_kernel", [&] {
    grid_cuda_kernel<scalar_t><<<BLOCKS(num_nodes), THREADS>>>(
        cluster.data<int64_t>(),
        at::cuda::detail::getTensorInfo<scalar_t, int>(pos),
        size.toType(pos.type()).data<scalar_t>(),
        start.toType(pos.type()).data<scalar_t>(),
        end.toType(pos.type()).data<scalar_t>(), num_nodes);
  });

  return cluster;
}
