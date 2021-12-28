#include "graclus_cuda.h"

#include <ATen/cuda/CUDAContext.h>

#include "utils.cuh"

#define THREADS 1024
#define BLOCKS(N) (N + THREADS - 1) / THREADS
#define BLUE_P 0.53406

__device__ bool done_d;
__global__ void init_done_kernel() { done_d = true; }
__global__ void colorize_kernel(int64_t *out, const float *bernoulli,
                                int64_t numel) {
  const int64_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (thread_idx < numel) {
    if (out[thread_idx] < 0) {
      out[thread_idx] = (int64_t)bernoulli[thread_idx] - 2;
      done_d = false;
    }
  }
}

bool colorize(torch::Tensor out) {
  auto stream = at::cuda::getCurrentCUDAStream();
  init_done_kernel<<<1, 1, 0, stream>>>();

  auto numel = out.size(0);
  auto props = torch::full(numel, BLUE_P, out.options().dtype(torch::kFloat));
  auto bernoulli = props.bernoulli();

  colorize_kernel<<<BLOCKS(numel), THREADS, 0, stream>>>(
      out.data_ptr<int64_t>(), bernoulli.data_ptr<float>(), numel);

  bool done_h;
  cudaMemcpyFromSymbol(&done_h, done_d, sizeof(done_h), 0,
                       cudaMemcpyDeviceToHost);
  return done_h;
}

__global__ void propose_kernel(int64_t *out, int64_t *proposal,
                               const int64_t *rowptr, const int64_t *col,
                               int64_t numel) {

  const int64_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (thread_idx < numel) {
    if (out[thread_idx] != -1)
      return; // Only vist blue nodes.

    bool has_unmatched_neighbor = false;

    for (int64_t i = rowptr[thread_idx]; i < rowptr[thread_idx + 1]; i++) {
      auto v = col[i];

      if (out[v] < 0)
        has_unmatched_neighbor = true; // Unmatched neighbor found.

      if (out[v] == -2) {
        proposal[thread_idx] = v; // Propose to first red neighbor.
        break;
      }
    }

    if (!has_unmatched_neighbor)
      out[thread_idx] = thread_idx;
  }
}

template <typename scalar_t>
__global__ void weighted_propose_kernel(int64_t *out, int64_t *proposal,
                                        const int64_t *rowptr,
                                        const int64_t *col,
                                        const scalar_t *weight, int64_t numel) {

  const int64_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (thread_idx < numel) {
    if (out[thread_idx] != -1)
      return; // Only vist blue nodes.

    bool has_unmatched_neighbor = false;
    int64_t v_max = -1;
    scalar_t w_max = 0;

    for (int64_t i = rowptr[thread_idx]; i < rowptr[thread_idx + 1]; i++) {
      auto v = col[i];

      if (out[v] < 0)
        has_unmatched_neighbor = true; // Unmatched neighbor found.

      // Find maximum weighted red neighbor.
      if (out[v] == -2 && weight[i] >= w_max) {
        v_max = v;
        w_max = weight[i];
      }
    }

    proposal[thread_idx] = v_max; // Propose.

    if (!has_unmatched_neighbor)
      out[thread_idx] = thread_idx;
  }
}

void propose(torch::Tensor out, torch::Tensor proposal, torch::Tensor rowptr,
             torch::Tensor col,
             torch::optional<torch::Tensor> optional_weight) {

  auto stream = at::cuda::getCurrentCUDAStream();

  if (!optional_weight.has_value()) {
    propose_kernel<<<BLOCKS(out.numel()), THREADS, 0, stream>>>(
        out.data_ptr<int64_t>(), proposal.data_ptr<int64_t>(),
        rowptr.data_ptr<int64_t>(), col.data_ptr<int64_t>(), out.numel());
  } else {
    auto weight = optional_weight.value();
    auto scalar_type = weight.scalar_type();
    AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, scalar_type, "_", [&] {
      weighted_propose_kernel<scalar_t>
          <<<BLOCKS(out.numel()), THREADS, 0, stream>>>(
              out.data_ptr<int64_t>(), proposal.data_ptr<int64_t>(),
              rowptr.data_ptr<int64_t>(), col.data_ptr<int64_t>(),
              weight.data_ptr<scalar_t>(), out.numel());
    });
  }
}

__global__ void respond_kernel(int64_t *out, const int64_t *proposal,
                               const int64_t *rowptr, const int64_t *col,
                               int64_t numel) {
  const int64_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (thread_idx < numel) {
    if (out[thread_idx] != -2)
      return; // Only vist red nodes.

    bool has_unmatched_neighbor = false;

    for (int64_t i = rowptr[thread_idx]; i < rowptr[thread_idx + 1]; i++) {
      auto v = col[i];

      if (out[v] < 0)
        has_unmatched_neighbor = true; // Unmatched neighbor found.

      if (out[v] == -1 && proposal[v] == thread_idx) {
        // Match first blue neighbhor v which proposed to u.
        out[thread_idx] = min(thread_idx, v);
        out[v] = min(thread_idx, v);
        break;
      }
    }

    if (!has_unmatched_neighbor)
      out[thread_idx] = thread_idx;
  }
}

template <typename scalar_t>
__global__ void weighted_respond_kernel(int64_t *out, const int64_t *proposal,
                                        const int64_t *rowptr,
                                        const int64_t *col,
                                        const scalar_t *weight, int64_t numel) {
  const int64_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (thread_idx < numel) {
    if (out[thread_idx] != -2)
      return; // Only vist red nodes.

    bool has_unmatched_neighbor = false;
    int64_t v_max = -1;
    scalar_t w_max = 0;

    for (int64_t i = rowptr[thread_idx]; i < rowptr[thread_idx + 1]; i++) {
      auto v = col[i];

      if (out[v] < 0)
        has_unmatched_neighbor = true; // Unmatched neighbor found.

      if (out[v] == -1 && proposal[v] == thread_idx && weight[i] >= w_max) {
        // Find maximum weighted blue neighbhor v which proposed to u.
        v_max = v;
        w_max = weight[i];
      }
    }

    if (v_max >= 0) {
      out[thread_idx] = min(thread_idx, v_max); // Match neighbors.
      out[v_max] = min(thread_idx, v_max);
    }

    if (!has_unmatched_neighbor)
      out[thread_idx] = thread_idx;
  }
}

void respond(torch::Tensor out, torch::Tensor proposal, torch::Tensor rowptr,
             torch::Tensor col,
             torch::optional<torch::Tensor> optional_weight) {

  auto stream = at::cuda::getCurrentCUDAStream();

  if (!optional_weight.has_value()) {
    respond_kernel<<<BLOCKS(out.numel()), THREADS, 0, stream>>>(
        out.data_ptr<int64_t>(), proposal.data_ptr<int64_t>(),
        rowptr.data_ptr<int64_t>(), col.data_ptr<int64_t>(), out.numel());
  } else {
    auto weight = optional_weight.value();
    auto scalar_type = weight.scalar_type();
    AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, scalar_type, "_", [&] {
      weighted_respond_kernel<scalar_t>
          <<<BLOCKS(out.numel()), THREADS, 0, stream>>>(
              out.data_ptr<int64_t>(), proposal.data_ptr<int64_t>(),
              rowptr.data_ptr<int64_t>(), col.data_ptr<int64_t>(),
              weight.data_ptr<scalar_t>(), out.numel());
    });
  }
}

torch::Tensor graclus_cuda(torch::Tensor rowptr, torch::Tensor col,
                           torch::optional<torch::Tensor> optional_weight) {
  CHECK_CUDA(rowptr);
  CHECK_CUDA(col);
  CHECK_INPUT(rowptr.dim() == 1 && col.dim() == 1);
  if (optional_weight.has_value()) {
    CHECK_CUDA(optional_weight.value());
    CHECK_INPUT(optional_weight.value().dim() == 1);
    CHECK_INPUT(optional_weight.value().numel() == col.numel());
  }
  cudaSetDevice(rowptr.get_device());

  int64_t num_nodes = rowptr.numel() - 1;
  auto out = torch::full(num_nodes, -1, rowptr.options());
  auto proposal = torch::full(num_nodes, -1, rowptr.options());

  while (!colorize(out)) {
    propose(out, proposal, rowptr, col, optional_weight);
    respond(out, proposal, rowptr, col, optional_weight);
  }

  return out;
}
