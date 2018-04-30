#include <ATen/ATen.h>
#include <ATen/cuda/detail/IndexUtils.cuh>

#define THREADS 1024
#define BLOCKS(N) (N + THREADS - 1) / THREADS

template <typename scalar_t>
__global__ void grid_cuda_kernel(
    int64_t *cluster, const at::cuda::detail::TensorInfo<scalar_t, int> pos,
    const scalar_t *__restrict__ size, const scalar_t *__restrict__ start,
    const scalar_t *__restrict__ end, const size_t num_nodes) {
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
  size = size.toType(pos.type());
  start = start.toType(pos.type());
  end = end.toType(pos.type());

  const auto num_nodes = pos.size(0);
  auto cluster = at::empty(pos.type().toScalarType(at::kLong), {num_nodes});

  AT_DISPATCH_ALL_TYPES(pos.type(), "grid_cuda_kernel", [&] {
    auto cluster_data = cluster.data<int64_t>();
    auto pos_info = at::cuda::detail::getTensorInfo<scalar_t, int>(pos);
    auto size_data = size.data<scalar_t>();
    auto start_data = start.data<scalar_t>();
    auto end_data = end.data<scalar_t>();
    grid_cuda_kernel<scalar_t><<<BLOCKS(num_nodes), THREADS>>>(
        cluster_data, pos_info, size_data, start_data, end_data, num_nodes);
  });

  return cluster;
}
