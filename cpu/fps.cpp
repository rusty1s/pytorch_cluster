#include <torch/extension.h>

#include "utils.h"

at::Tensor get_dist(at::Tensor x, ptrdiff_t index) {
  return (x - x[index]).norm(2, 1);
}

at::Tensor fps(at::Tensor x, at::Tensor batch, float ratio, bool random) {
  auto batch_size = batch[-1].data<int64_t>()[0] + 1;

  auto deg = degree(batch, batch_size);
  auto cum_deg = at::cat({at::zeros(1, deg.options()), deg.cumsum(0)}, 0);
  auto k = (deg.toType(at::kFloat) * ratio).ceil().toType(at::kLong);
  auto cum_k = at::cat({at::zeros(1, k.options()), k.cumsum(0)}, 0);

  auto out = at::empty(cum_k[-1].data<int64_t>()[0], batch.options());

  auto cum_deg_d = cum_deg.data<int64_t>();
  auto k_d = k.data<int64_t>();
  auto cum_k_d = cum_k.data<int64_t>();
  auto out_d = out.data<int64_t>();

  for (ptrdiff_t b = 0; b < batch_size; b++) {
    auto index = at::range(cum_deg_d[b], cum_deg_d[b + 1] - 1, out.options());
    auto y = x.index_select(0, index);

    ptrdiff_t start = 0;
    if (random) {
      start = at::randperm(y.size(0), batch.options()).data<int64_t>()[0];
    }

    out_d[cum_k_d[b]] = cum_deg_d[b] + start;
    auto dist = get_dist(y, start);

    for (ptrdiff_t i = 1; i < k_d[b]; i++) {
      ptrdiff_t argmax = dist.argmax().data<int64_t>()[0];
      out_d[cum_k_d[b] + i] = cum_deg_d[b] + argmax;
      dist = at::min(dist, get_dist(y, argmax));
    }
  }

  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fps", &fps, "Farthest Point Sampling (CPU)");
}
