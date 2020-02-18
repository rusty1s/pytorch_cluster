#include "graclus_cpu.h"

#include "utils.h"

torch::Tensor graclus_cpu(torch::Tensor row, torch::Tensor col,
                          torch::optional<torch::Tensor> optional_weight,
                          int64_t num_nodes) {

  CHECK_CPU(row);
  CHECK_CPU(col);
  CHECK_INPUT(row.dim() == 1 && col.dim() == 1 && row.numel() == col.numel());
  if (optional_weight.has_value()) {
    CHECK_CPU(optional_weight.value());
    CHECK_INPUT(optional_weight.value().numel() == col.numel());
  }

  auto mask = row != col;
  row = row.masked_select(mask), col = col.masked_select(mask);
  if (optional_weight.has_value())
    optional_weight = optional_weight.value().masked_select(mask);

  auto perm = torch::randperm(row.size(0), row.options());
  row = row.index_select(0, perm);
  col = col.index_select(0, perm);
  if (optional_weight.has_value())
    optional_weight = optional_weight.value().index_select(0, perm);

  std::tie(row, perm) = row.sort();
  col = col.index_select(0, perm);
  if (optional_weight.has_value())
    optional_weight = optional_weight.value().index_select(0, perm);

  auto rowptr = torch::zeros(num_nodes, row.options());
  rowptr = rowptr.scatter_add_(0, row, torch::ones_like(row)).cumsum(0);
  rowptr = torch::cat({torch::zeros(1, row.options()), rowptr}, 0);

  perm = torch::randperm(num_nodes, row.options());
  auto out = torch::full(num_nodes, -1, row.options());

  auto rowptr_data = rowptr.data_ptr<int64_t>();
  auto col_data = col.data_ptr<int64_t>();
  auto perm_data = perm.data_ptr<int64_t>();
  auto out_data = out.data_ptr<int64_t>();

  if (!optional_weight.has_value()) {
    for (auto i = 0; i < num_nodes; i++) {
      auto u = perm_data[i];

      if (out_data[u] >= 0)
        continue;

      out_data[u] = u;

      for (auto j = rowptr_data[u]; j < rowptr_data[u + 1]; j++) {
        auto v = col_data[j];

        if (out_data[v] >= 0)
          continue;

        out_data[u] = std::min(u, v);
        out_data[v] = std::min(u, v);
        break;
      }
    }
  } else {
    auto weight = optional_weight.value();
    AT_DISPATCH_ALL_TYPES(weight.scalar_type(), "weighted_graclus", [&] {
      auto weight_data = weight.data_ptr<scalar_t>();

      for (auto i = 0; i < num_nodes; i++) {
        auto u = perm_data[i];

        if (out_data[u] >= 0)
          continue;

        auto v_max = u;
        scalar_t w_max = (scalar_t)0.;

        for (auto j = rowptr_data[u]; j < rowptr_data[u + 1]; j++) {
          auto v = col_data[j];

          if (out_data[v] >= 0)
            continue;

          if (weight_data[j] >= w_max) {
            v_max = v;
            w_max = weight_data[j];
          }
        }

        out_data[u] = std::min(u, v_max);
        out_data[v_max] = std::min(u, v_max);
      }
    });
  }

  return out;
}
