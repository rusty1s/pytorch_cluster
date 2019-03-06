#include <ATen/ATen.h>

#include "coloring.cuh"
#include "proposal.cuh"
#include "response.cuh"
#include "utils.cuh"

at::Tensor graclus_cuda(at::Tensor row, at::Tensor col, int64_t num_nodes) {
  cudaSetDevice(row.get_device());
  std::tie(row, col) = remove_self_loops(row, col);
  std::tie(row, col) = rand(row, col);
  std::tie(row, col) = to_csr(row, col, num_nodes);

  auto cluster = at::full(num_nodes, -1, row.options());
  auto proposal = at::full(num_nodes, -1, row.options());

  while (!colorize(cluster)) {
    propose(cluster, proposal, row, col);
    respond(cluster, proposal, row, col);
  }

  return cluster;
}

at::Tensor weighted_graclus_cuda(at::Tensor row, at::Tensor col,
                                 at::Tensor weight, int64_t num_nodes) {
  cudaSetDevice(row.get_device());
  std::tie(row, col, weight) = remove_self_loops(row, col, weight);
  std::tie(row, col, weight) = to_csr(row, col, weight, num_nodes);

  auto cluster = at::full(num_nodes, -1, row.options());
  auto proposal = at::full(num_nodes, -1, row.options());

  while (!colorize(cluster)) {
    propose(cluster, proposal, row, col, weight);
    respond(cluster, proposal, row, col, weight);
  }

  return cluster;
}
