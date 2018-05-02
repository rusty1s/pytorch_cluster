#include <torch/torch.h>

#include "../include/degree.cpp"
#include "../include/loop.cpp"
#include "../include/perm.cpp"

at::Tensor graclus(at::Tensor row, at::Tensor col, int num_nodes) {
  std::tie(row, col) = remove_self_loops(row, col);
  std::tie(row, col) = randperm(row, col, num_nodes);
  auto deg = degree(row, num_nodes, row.type().scalarType());

  auto cluster = at::full(row.type(), {num_nodes}, -1);

  auto *row_data = row.data<int64_t>();
  auto *col_data = col.data<int64_t>();
  auto *deg_data = deg.data<int64_t>();
  auto *cluster_data = cluster.data<int64_t>();

  int64_t e_idx = 0, d_idx, r, c;
  while (e_idx < row.size(0)) {
    r = row_data[e_idx];
    if (cluster_data[r] < 0) {
      cluster_data[r] = r;
      for (d_idx = 0; d_idx < deg_data[r]; d_idx++) {
        c = col_data[e_idx + d_idx];
        if (cluster_data[c] < 0) {
          cluster_data[r] = std::min(r, c);
          cluster_data[c] = std::min(r, c);
          break;
        }
      }
    }
    e_idx += deg_data[r];
  }

  return cluster;
}
