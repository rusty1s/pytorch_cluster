#include "fps_cpu.h"

#include "utils.h"

inline torch::Tensor get_dist(torch::Tensor x, int64_t idx) {
  return (x - x[idx]).norm(2, 1);
}

torch::Tensor fps_cpu(torch::Tensor src,
                      torch::optional<torch::Tensor> optional_ptr, double ratio,
                      bool random_start) {

  CHECK_CPU(src);
  if (optional_ptr.has_value()) {
    CHECK_CPU(optional_ptr.value());
    CHECK_INPUT(optional_ptr.value().dim() == 1);
  }
  AT_ASSERTM(ratio > 0 and ratio < 1, "Invalid input");

  if (!optional_ptr.has_value())
    optional_ptr =
        torch::tensor({0, src.size(0)}, src.options().dtype(torch::kLong));

  src = src.view({src.size(0), -1}).contiguous();
  auto ptr = optional_ptr.value().contiguous();
  auto batch_size = ptr.size(0) - 1;

  auto deg = ptr.narrow(0, 1, batch_size) - ptr.narrow(0, 0, batch_size);
  auto out_ptr = deg.toType(torch::kFloat) * (float)ratio;
  out_ptr = out_ptr.ceil().toType(torch::kLong).cumsum(0);

  auto out = torch::empty(out_ptr[-1].data_ptr<int64_t>()[0], ptr.options());

  auto ptr_data = ptr.data_ptr<int64_t>();
  auto out_ptr_data = out_ptr.data_ptr<int64_t>();
  auto out_data = out.data_ptr<int64_t>();

  int64_t src_start = 0, out_start = 0, src_end, out_end;
  for (auto b = 0; b < batch_size; b++) {
    src_end = ptr_data[b + 1], out_end = out_ptr_data[b];
    auto y = src.narrow(0, src_start, src_end - src_start);

    int64_t start_idx = 0;
    if (random_start) {
      // TODO: GET RANDOM INTEGER
    }

    out_data[out_start] = src_start + start_idx;
    auto dist = get_dist(y, start_idx);

    for (auto i = 1; i < out_end - out_start; i++) {
      int64_t argmax = dist.argmax().data_ptr<int64_t>()[0];
      out_data[out_start + i] = src_start + argmax;
      dist = torch::min(dist, get_dist(y, argmax));
    }

    src_start = src_end, out_start = out_end;
  }
  return out;
}
