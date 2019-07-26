#include <torch/extension.h>

#include "utils.h"

at::Tensor rw(at::Tensor row, at::Tensor col, at::Tensor start,
              size_t walk_length, float p, float q, size_t num_nodes) {
  auto deg = degree(row, num_nodes);
  auto cum_deg = at::cat({at::zeros(1, deg.options()), deg.cumsum(0)}, 0);

  auto rand = at::rand({start.size(0), (int64_t)walk_length},
                       start.options().dtype(at::kFloat));
  auto out =
      at::full({start.size(0), (int64_t)walk_length + 1}, -1, start.options());

  auto deg_d = deg.data<int64_t>();
  auto cum_deg_d = cum_deg.data<int64_t>();
  auto col_d = col.data<int64_t>();
  auto start_d = start.data<int64_t>();
  auto rand_d = rand.data<float>();
  auto out_d = out.data<int64_t>();

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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rw", &rw, "Random Walk Sampling (CPU)");
}
