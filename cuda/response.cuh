#pragma once

#include <ATen/ATen.h>

#define THREADS 1024
#define BLOCKS(N) (N + THREADS - 1) / THREADS

__global__ void respond_kernel(int64_t *__restrict__ cluster, int64_t *proposal,
                               int64_t *__restrict row,
                               int64_t *__restrict__ col, size_t numel) {
  const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t stride = blockDim.x * gridDim.x;
  for (int64_t u = index; u < numel; u += stride) {
    if (cluster[u] != -2)
      continue; // Only vist red nodes.

    bool has_unmatched_neighbor = false;

    for (int64_t i = row[u]; i < row[u + 1]; i++) {
      auto v = col[i];

      if (cluster[v] < 0)
        has_unmatched_neighbor = true; // Unmatched neighbor found.

      if (cluster[v] == -1 && proposal[v] == u) {
        // Match first blue neighbhor v which proposed to u.
        cluster[u] = min(u, v);
        cluster[v] = min(u, v);
        break;
      }
    }

    if (!has_unmatched_neighbor)
      cluster[u] = u;
  }
}

void respond(at::Tensor cluster, at::Tensor proposal, at::Tensor row,
             at::Tensor col) {
  respond_kernel<<<BLOCKS(cluster.numel()), THREADS>>>(
      cluster.data<int64_t>(), proposal.data<int64_t>(), row.data<int64_t>(),
      col.data<int64_t>(), cluster.numel());
}

template <typename scalar_t>
__global__ void respond_kernel(int64_t *__restrict__ cluster, int64_t *proposal,
                               int64_t *__restrict row,
                               int64_t *__restrict__ col,
                               scalar_t *__restrict__ weight, size_t numel) {
  const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t stride = blockDim.x * gridDim.x;
  for (int64_t u = index; u < numel; u += stride) {
    if (cluster[u] != -2)
      continue; // Only vist red nodes.

    bool has_unmatched_neighbor = false;
    int64_t v_max = -1;
    scalar_t w_max = 0;

    for (int64_t i = row[u]; i < row[u + 1]; i++) {
      auto v = col[i];

      if (cluster[v] < 0)
        has_unmatched_neighbor = true; // Unmatched neighbor found.

      if (cluster[v] == -1 && proposal[v] == u && weight[i] >= w_max) {
        // Find maximum weighted blue neighbhor v which proposed to u.
        v_max = v;
        w_max = weight[i];
      }
    }

    if (v_max >= 0) {
      cluster[u] = min(u, v_max); // Match neighbors.
      cluster[v_max] = min(u, v_max);
    }

    if (!has_unmatched_neighbor)
      cluster[u] = u;
  }
}

void respond(at::Tensor cluster, at::Tensor proposal, at::Tensor row,
             at::Tensor col, at::Tensor weight) {
  AT_DISPATCH_ALL_TYPES(weight.scalar_type(), "respond_kernel", [&] {
    respond_kernel<scalar_t><<<BLOCKS(cluster.numel()), THREADS>>>(
        cluster.data<int64_t>(), proposal.data<int64_t>(), row.data<int64_t>(),
        col.data<int64_t>(), weight.data<scalar_t>(), cluster.numel());
  });
}
