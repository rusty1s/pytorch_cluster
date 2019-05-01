#pragma once

#include <ATen/ATen.h>

#define THREADS 1024
#define BLOCKS(N) (N + THREADS - 1) / THREADS

__global__ void propose_kernel(int64_t *__restrict__ cluster, int64_t *proposal,
                               int64_t *__restrict row,
                               int64_t *__restrict__ col, size_t numel) {
  const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t stride = blockDim.x * gridDim.x;
  for (int64_t u = index; u < numel; u += stride) {
    if (cluster[u] != -1)
      continue; // Only vist blue nodes.

    bool has_unmatched_neighbor = false;

    for (int64_t i = row[u]; i < row[u + 1]; i++) {
      auto v = col[i];

      if (cluster[v] < 0)
        has_unmatched_neighbor = true; // Unmatched neighbor found.

      if (cluster[v] == -2) {
        proposal[u] = v; // Propose to first red neighbor.
        break;
      }
    }

    if (!has_unmatched_neighbor)
      cluster[u] = u;
  }
}

void propose(at::Tensor cluster, at::Tensor proposal, at::Tensor row,
             at::Tensor col) {
  propose_kernel<<<BLOCKS(cluster.numel()), THREADS>>>(
      cluster.data<int64_t>(), proposal.data<int64_t>(), row.data<int64_t>(),
      col.data<int64_t>(), cluster.numel());
}

template <typename scalar_t>
__global__ void propose_kernel(int64_t *__restrict__ cluster, int64_t *proposal,
                               int64_t *__restrict row,
                               int64_t *__restrict__ col,
                               scalar_t *__restrict__ weight, size_t numel) {
  const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t stride = blockDim.x * gridDim.x;
  for (int64_t u = index; u < numel; u += stride) {
    if (cluster[u] != -1)
      continue; // Only vist blue nodes.

    bool has_unmatched_neighbor = false;
    int64_t v_max = -1;
    scalar_t w_max = 0;

    for (int64_t i = row[u]; i < row[u + 1]; i++) {
      auto v = col[i];

      if (cluster[v] < 0)
        has_unmatched_neighbor = true; // Unmatched neighbor found.

      // Find maximum weighted red neighbor.
      if (cluster[v] == -2 && weight[i] >= w_max) {
        v_max = v;
        w_max = weight[i];
      }
    }

    proposal[u] = v_max; // Propose.

    if (!has_unmatched_neighbor)
      cluster[u] = u;
  }
}

void propose(at::Tensor cluster, at::Tensor proposal, at::Tensor row,
             at::Tensor col, at::Tensor weight) {
  AT_DISPATCH_ALL_TYPES(weight.scalar_type(), "propose_kernel", [&] {
    propose_kernel<scalar_t><<<BLOCKS(cluster.numel()), THREADS>>>(
        cluster.data<int64_t>(), proposal.data<int64_t>(), row.data<int64_t>(),
        col.data<int64_t>(), weight.data<scalar_t>(), cluster.numel());
  });
}
