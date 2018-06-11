#pragma once

#include <ATen/ATen.h>

#define THREADS 1024
#define BLOCKS(N) (N + THREADS - 1) / THREADS

inline at::Tensor degree(at::Tensor index, int num_nodes) {
  auto zero = at::zeros(index.type(), {num_nodes});
  auto one = at::ones(index.type(), {index.size(0)});
  return zero.scatter_add_(0, index, one);
}
