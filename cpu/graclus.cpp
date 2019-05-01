#include <torch/extension.h>

#include "utils.h"

at::Tensor graclus(at::Tensor row, at::Tensor col, int64_t num_nodes) {
  std::tie(row, col) = remove_self_loops(row, col);
  std::tie(row, col) = rand(row, col);
  std::tie(row, col) = to_csr(row, col, num_nodes);
  auto row_data = row.data<int64_t>(), col_data = col.data<int64_t>();

  auto perm = at::randperm(num_nodes, row.options());
  auto perm_data = perm.data<int64_t>();

  auto cluster = at::full(num_nodes, -1, row.options());
  auto cluster_data = cluster.data<int64_t>();

  for (int64_t i = 0; i < num_nodes; i++) {
    auto u = perm_data[i];

    if (cluster_data[u] >= 0)
      continue;

    cluster_data[u] = u;

    for (int64_t j = row_data[u]; j < row_data[u + 1]; j++) {
      auto v = col_data[j];

      if (cluster_data[v] >= 0)
        continue;

      cluster_data[u] = std::min(u, v);
      cluster_data[v] = std::min(u, v);
      break;
    }
  }

  return cluster;
}

at::Tensor weighted_graclus(at::Tensor row, at::Tensor col, at::Tensor weight,
                            int64_t num_nodes) {
  std::tie(row, col, weight) = remove_self_loops(row, col, weight);
  std::tie(row, col, weight) = to_csr(row, col, weight, num_nodes);
  auto row_data = row.data<int64_t>(), col_data = col.data<int64_t>();

  auto perm = at::randperm(num_nodes, row.options());
  auto perm_data = perm.data<int64_t>();

  auto cluster = at::full(num_nodes, -1, row.options());
  auto cluster_data = cluster.data<int64_t>();

  AT_DISPATCH_ALL_TYPES(weight.scalar_type(), "weighted_graclus", [&] {
    auto weight_data = weight.data<scalar_t>();

    for (int64_t i = 0; i < num_nodes; i++) {
      auto u = perm_data[i];

      if (cluster_data[u] >= 0)
        continue;

      int64_t v_max = u;
      scalar_t w_max = 0;

      for (int64_t j = row_data[u]; j < row_data[u + 1]; j++) {
        auto v = col_data[j];

        if (cluster_data[v] >= 0)
          continue;

        if (weight_data[j] >= w_max) {
          v_max = v;
          w_max = weight_data[j];
        }
      }

      cluster_data[u] = std::min(u, v_max);
      cluster_data[v_max] = std::min(u, v_max);
    }
  });

  return cluster;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("graclus", &graclus, "Graclus (CPU)");
  m.def("weighted_graclus", &weighted_graclus, "Weighted Graclus (CPU)");
}
