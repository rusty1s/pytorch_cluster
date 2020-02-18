#include "rw_cpu.h"

#include "utils.h"

torch::Tensor random_walk_cpu(torch::Tensor rowptr, torch::Tensor col,
                              torch::Tensor start, int64_t walk_length,
                              double p, double q) {
  CHECK_CPU(rowptr);
  CHECK_CPU(col);
  CHECK_CPU(start);

  CHECK_INPUT(rowptr.dim() == 1);
  CHECK_INPUT(col.dim() == 1);
  CHECK_INPUT(start.dim() == 1);

  auto num_nodes = rowptr.size(0) - 1;
  auto deg = rowptr.narrow(0, 1, num_nodes) - rowptr.narrow(0, 0, num_nodes);

  auto rand = torch::rand({start.size(0), walk_length},
                          start.options().dtype(torch::kFloat));

  auto out = torch::full({start.size(0), walk_length + 1}, -1, start.options());

  auto rowptr_data = rowptr.data_ptr<int64_t>();
  auto deg_data = deg.data_ptr<int64_t>();
  auto col_data = col.data_ptr<int64_t>();
  auto start_data = start.data_ptr<int64_t>();
  auto rand_data = rand.data_ptr<float>();
  auto out_data = out.data_ptr<int64_t>();

  for (auto n = 0; n < start.size(0); n++) {
    auto cur = start_data[n];
    auto offset = n * (walk_length + 1);
    out_data[offset] = cur;

    for (auto l = 1; l <= walk_length; l++) {
      cur = col_data[rowptr_data[cur] +
                     int64_t(rand_data[n * walk_length + (l - 1)] *
                             deg_data[cur])];
      out_data[offset + l] = cur;
    }
  }

  return out;
}
