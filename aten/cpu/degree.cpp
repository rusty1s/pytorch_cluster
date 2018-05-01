#ifndef DEGREE_CPP
#define DEGREE_CPP

#include <torch/torch.h>

inline at::Tensor degree(at::Tensor index, int num_nodes,
                         at::ScalarType scalar_type) {
  auto zero = at::full(torch::CPU(scalar_type), {num_nodes}, 0);
  auto one = at::full(zero.type(), {index.size(0)}, 1);
  return zero.scatter_add_(0, index, one);
}

#endif // DEGREE_CPP
