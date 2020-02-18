#include "rw_cpu.h"

#include "utils.h"

at::Tensor random_walk_cpu(torch::Tensor row, torch::Tensor col,
                           torch::Tensor start, int64_t walk_length, double p,
                           double q, int64_t num_nodes) {

  auto deg = degree(row, num_nodes);
  auto cum_deg = at::cat({at::zeros(1, deg.options()), deg.cumsum(0)}, 0);

  auto rand = at::rand({start.size(0), (int64_t)walk_length},
                       start.options().dtype(at::kFloat));
  auto out =
      at::full({start.size(0), (int64_t)walk_length + 1}, -1, start.options());

  auto deg_d = deg.DATA_PTR<int64_t>();
  auto cum_deg_d = cum_deg.DATA_PTR<int64_t>();
  auto col_d = col.DATA_PTR<int64_t>();
  auto start_d = start.DATA_PTR<int64_t>();
  auto rand_d = rand.DATA_PTR<float>();
  auto out_d = out.DATA_PTR<int64_t>();

  for (ptrdiff_t n = 0; n < start.size(0); n++) {
    int64_t cur = start_d[n];
    auto i = n * (walk_length + 1);
    out_d[i] = cur;

    for (ptrdiff_t l = 1; l <= (int64_t)walk_length; l++) {
      cur = col_d[cum_deg_d[cur] +
                  int64_t(rand_d[n * walk_length + (l - 1)] * deg_d[cur])];
      out_d[i + l] = cur;
    }
  }

  return out;
}
