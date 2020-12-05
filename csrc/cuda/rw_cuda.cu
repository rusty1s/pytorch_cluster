#include "rw_cuda.h"

#include <ATen/cuda/CUDAContext.h>
#include <curand.h>
#include <curand_kernel.h>

#include "utils.cuh"

#define THREADS 1024
#define BLOCKS(N) (N + THREADS - 1) / THREADS

__global__ void uniform_sampling_kernel(const int64_t *rowptr,
                                        const int64_t *col,
                                        const int64_t *start, const float *rand,
                                        int64_t *n_out, int64_t *e_out,
                                        const int64_t walk_length,
                                        const int64_t numel) {

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

__global__ void
rejection_sampling_kernel(unsigned int seed, const int64_t *rowptr,
                          const int64_t *col, const int64_t *start,
                          int64_t *n_out, int64_t *e_out,
                          const int64_t walk_length, const int64_t numel,
                          const double p, const double q) {

  curandState_t state;
  curand_init(seed, 0, 0, &state);

  double max_prob = fmax(fmax(1. / p, 1.), 1. / q);
  double prob_0 = 1. / p / max_prob;
  double prob_1 = 1. / max_prob;
  double prob_2 = 1. / q / max_prob;

  const int64_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (thread_idx < numel) {
    int64_t t = start[thread_idx], v, x, e_cur, row_start, row_end;

    n_out[thread_idx] = t;

    row_start = rowptr[t], row_end = rowptr[t + 1];
    if (row_end - row_start == 0) {
      e_cur = -1;
      v = t;
    } else {
      e_cur = row_start + (curand(&state) % (row_end - row_start));
      v = col[e_cur];
    }

    n_out[numel + thread_idx] = v;
    e_out[thread_idx] = e_cur;

    for (int64_t l = 1; l < walk_length; l++) {
      row_start = rowptr[v], row_end = rowptr[v + 1];

      if (row_end - row_start == 0) {
        e_cur = -1;
        x = v;
      } else if (row_end - row_start == 1) {
        e_cur = row_start;
        x = col[e_cur];
      } else {
        while (true) {
          e_cur = row_start + (curand(&state) % (row_end - row_start));
          x = col[e_cur];

          double r = curand_uniform(&state); // (0, 1]

          if (x == t && r < prob_0)
            break;

          bool is_neighbor = false;
          row_start = rowptr[x], row_end = rowptr[x + 1];
          for (int64_t i = row_start; i < row_end; i++) {
            if (col[i] == t) {
              is_neighbor = true;
              break;
            }
          }

          if (is_neighbor && r < prob_1)
            break;
          else if (r < prob_2)
            break;
        }
      }

      n_out[(l + 1) * numel + thread_idx] = x;
      e_out[l * numel + thread_idx] = e_cur;
      t = v;
      v = x;
    }
  }
}

std::tuple<torch::Tensor, torch::Tensor>
random_walk_cuda(torch::Tensor rowptr, torch::Tensor col, torch::Tensor start,
                 int64_t walk_length, double p, double q) {
  CHECK_CUDA(rowptr);
  CHECK_CUDA(col);
  CHECK_CUDA(start);
  cudaSetDevice(rowptr.get_device());

  CHECK_INPUT(rowptr.dim() == 1);
  CHECK_INPUT(col.dim() == 1);
  CHECK_INPUT(start.dim() == 1);

  auto n_out = torch::empty({walk_length + 1, start.size(0)}, start.options());
  auto e_out = torch::empty({walk_length, start.size(0)}, start.options());

  auto stream = at::cuda::getCurrentCUDAStream();

  if (p == 1. && q == 1.) {
    auto rand = torch::rand({start.size(0), walk_length},
                            start.options().dtype(torch::kFloat));

    uniform_sampling_kernel<<<BLOCKS(start.numel()), THREADS, 0, stream>>>(
        rowptr.data_ptr<int64_t>(), col.data_ptr<int64_t>(),
        start.data_ptr<int64_t>(), rand.data_ptr<float>(),
        n_out.data_ptr<int64_t>(), e_out.data_ptr<int64_t>(), walk_length,
        start.numel());
  } else {
    rejection_sampling_kernel<<<BLOCKS(start.numel()), THREADS, 0, stream>>>(
        time(NULL), rowptr.data_ptr<int64_t>(), col.data_ptr<int64_t>(),
        start.data_ptr<int64_t>(), n_out.data_ptr<int64_t>(),
        e_out.data_ptr<int64_t>(), walk_length, start.numel(), p, q);
  }

  return std::make_tuple(n_out.t().contiguous(), e_out.t().contiguous());
}
