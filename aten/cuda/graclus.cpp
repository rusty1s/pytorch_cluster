#include <torch/torch.h>

#include "../include/degree.cpp"
#include "../include/loop.cpp"

at::Tensor graclus(at::Tensor row, at::Tensor col, int num_nodes) {
  CHECK_CUDA(row);
  CHECK_CUDA(col);

  std::tie(row, col) = remove_self_loops(row, col);
  auto deg = degree(row, num_nodes, row.type().scalarType());

  return deg;
}
