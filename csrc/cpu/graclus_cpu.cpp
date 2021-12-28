#include "graclus_cpu.h"

#include "utils.h"

torch::Tensor graclus_cpu(torch::Tensor rowptr, torch::Tensor col,
                          torch::optional<torch::Tensor> optional_weight) {
  CHECK_CPU(rowptr);
  CHECK_CPU(col);
  CHECK_INPUT(rowptr.dim() == 1 && col.dim() == 1);
  if (optional_weight.has_value()) {
    CHECK_CPU(optional_weight.value());
    CHECK_INPUT(optional_weight.value().dim() == 1);
    CHECK_INPUT(optional_weight.value().numel() == col.numel());
  }

  int64_t num_nodes = rowptr.numel() - 1;
  auto out = torch::full(num_nodes, -1, rowptr.options());
  auto node_perm = torch::randperm(num_nodes, rowptr.options());

  auto rowptr_data = rowptr.data_ptr<int64_t>();
  auto col_data = col.data_ptr<int64_t>();
  auto node_perm_data = node_perm.data_ptr<int64_t>();
  auto out_data = out.data_ptr<int64_t>();

  if (!optional_weight.has_value()) {
    for (int64_t n = 0; n < num_nodes; n++) {
      auto u = node_perm_data[n];

      if (out_data[u] >= 0)
        continue;

      out_data[u] = u;

      int64_t row_start = rowptr_data[u], row_end = rowptr_data[u + 1];

      for (auto e = 0; e < row_end - row_start; e++) {
        auto v = col_data[row_start + e];

        if (out_data[v] >= 0)
          continue;

        out_data[u] = std::min(u, v);
        out_data[v] = std::min(u, v);
        break;
      }
    }
  } else {
    auto weight = optional_weight.value();
    auto scalar_type = weight.scalar_type();
    AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, scalar_type, "_", [&] {
      auto weight_data = weight.data_ptr<scalar_t>();

      for (auto n = 0; n < num_nodes; n++) {
        auto u = node_perm_data[n];

        if (out_data[u] >= 0)
          continue;

        auto v_max = u;
        scalar_t w_max = (scalar_t)0.;

        for (auto e = rowptr_data[u]; e < rowptr_data[u + 1]; e++) {
          auto v = col_data[e];

          if (out_data[v] >= 0)
            continue;

          if (weight_data[e] >= w_max) {
            v_max = v;
            w_max = weight_data[e];
          }
        }

        out_data[u] = std::min(u, v_max);
        out_data[v_max] = std::min(u, v_max);
      }
    });
  }

  return out;
}
