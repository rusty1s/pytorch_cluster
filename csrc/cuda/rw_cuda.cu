#include "rw_cuda.h"

#include <ATen/cuda/CUDAContext.h>

#include "utils.cuh"

#define THREADS 1024
#define BLOCKS(N) (N + THREADS - 1) / THREADS

__global__ void uniform_random_walk_kernel(const int64_t *rowptr,
                                           const int64_t *col,
                                           const int64_t *start,
                                           const float *rand, int64_t *n_out,
                                           int64_t *e_out, int64_t walk_length,
                                           int64_t numel) {
  const int64_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (thread_idx < numel) {
    int64_t n_cur = start[thread_idx], e_cur, row_start, row_end, rnd;

    n_out[thread_idx] = n_cur;

    for (int64_t l = 0; l < walk_length; l++) {
      row_start = rowptr[n_cur], row_end = rowptr[n_cur + 1];
      if (row_end - row_start == 0) {
        e_cur = -1;
      } else {
        rnd = int64_t(rand[l * numel + thread_idx] * (row_end - row_start));
        e_cur = row_start + rnd;
        n_cur = col[e_cur];
      }
      n_out[(l + 1) * numel + thread_idx] = n_cur;
      e_out[l * numel + thread_idx] = e_cur;
    }
  }
}

torch::Tensor random_walk_cuda(torch::Tensor rowptr, torch::Tensor col,
                               torch::Tensor start, int64_t walk_length,
                               double p, double q) {
  CHECK_CUDA(rowptr);
  CHECK_CUDA(col);
  CHECK_CUDA(start);
  cudaSetDevice(rowptr.get_device());

  CHECK_INPUT(rowptr.dim() == 1);
  CHECK_INPUT(col.dim() == 1);
  CHECK_INPUT(start.dim() == 1);

  auto rand = torch::rand({start.size(0), walk_length},
                          start.options().dtype(torch::kFloat));

  auto n_out = torch::empty({walk_length + 1, start.size(0)}, start.options());
  auto e_out = torch::empty({walk_length, start.size(0)}, start.options());

  auto stream = at::cuda::getCurrentCUDAStream();
  uniform_random_walk_kernel<<<BLOCKS(start.numel()), THREADS, 0, stream>>>(
      rowptr.data_ptr<int64_t>(), col.data_ptr<int64_t>(),
      start.data_ptr<int64_t>(), rand.data_ptr<float>(),
      n_out.data_ptr<int64_t>(), e_out.data_ptr<int64_t>(), walk_length,
      start.numel());

  return n_out.t().contiguous();
}
