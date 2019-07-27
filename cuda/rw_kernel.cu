#include <ATen/ATen.h>

#include "utils.cuh"

#define THREADS 1024
#define BLOCKS(N) (N + THREADS - 1) / THREADS

__global__ void uniform_rw_kernel(
    const int64_t *__restrict__ row, const int64_t *__restrict__ col,
    const int64_t *__restrict__ deg, const int64_t *__restrict__ start,
    const float *__restrict__ rand, int64_t *__restrict__ out,
    const size_t walk_length, const size_t numel) {

  const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t stride = blockDim.x * gridDim.x;

  for (ptrdiff_t n = index; n < numel; n += stride) {
    out[n] = start[n];

    for (ptrdiff_t l = 1; l <= walk_length; l++) {
      auto i = (l - 1) * numel + n;
      auto cur = out[i];
      out[l * numel + n] = col[row[cur] + int64_t(rand[i] * deg[cur])];
    }
  }
}

at::Tensor rw_cuda(at::Tensor row, at::Tensor col, at::Tensor start,
                   size_t walk_length, float p, float q, size_t num_nodes) {
  cudaSetDevice(row.get_device());
  auto deg = degree(row, num_nodes);
  row = at::cat({at::zeros(1, deg.options()), deg.cumsum(0)}, 0);

  auto rand = at::rand({(int64_t)walk_length, start.size(0)},
                       start.options().dtype(at::kFloat));
  auto out =
      at::full({(int64_t)walk_length + 1, start.size(0)}, -1, start.options());

  uniform_rw_kernel<<<BLOCKS(start.numel()), THREADS>>>(
      row.data<int64_t>(), col.data<int64_t>(), deg.data<int64_t>(),
      start.data<int64_t>(), rand.data<float>(), out.data<int64_t>(),
      walk_length, start.numel());

  return out.t().contiguous();
}
