#pragma once

#include <ATen/ATen.h>

#define THREADS 1024
#define BLOCKS(N) (N + THREADS - 1) / THREADS

#define BLUE_PROB 0.53406

__device__ int64_t done;

__global__ void init_done_kernel() { done = 1; }

__global__ void colorize_kernel(int64_t *cluster, float *__restrict__ bernoulli,
                                size_t numel) {
  const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t stride = blockDim.x * gridDim.x;
  for (int64_t u = index; u < numel; u += stride) {
    if (cluster[u] < 0) {
      cluster[u] = (int64_t)bernoulli[u] - 2;
      done = 0;
    }
  }
}

int64_t colorize(at::Tensor cluster) {
  init_done_kernel<<<1, 1>>>();

  auto numel = cluster.size(0);
  auto props = at::full(numel, BLUE_PROB, cluster.options().dtype(at::kFloat));
  auto bernoulli = props.bernoulli();

  colorize_kernel<<<BLOCKS(numel), THREADS>>>(cluster.data<int64_t>(),
                                              bernoulli.data<float>(), numel);

  int64_t out;
  cudaMemcpyFromSymbol(&out, done, sizeof(out), 0, cudaMemcpyDeviceToHost);
  return out;
}
