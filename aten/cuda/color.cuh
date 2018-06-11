#pragma once

#include <ATen/ATen.h>

#include "common.cuh"

#define BLUE_PROB 0.53406

__global__ void color_kernel(int64_t *cluster, size_t num_nodes) {
  const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t stride = blockDim.x * gridDim.x;
  for (ptrdiff_t i = index; i < num_nodes; i += stride) {
  }
}

inline bool color(at::Tensor cluster) {
  color_kernel<scalar_t><<<BLOCKS(cluster.size(0)), THREADS>>>(
      cluster.data<int64_t>(), cluster.size(0));

  return true;
}
