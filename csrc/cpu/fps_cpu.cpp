#include "fps_cpu.h"

#include <ATen/Parallel.h>

#include "utils.h"

inline torch::Tensor get_dist(torch::Tensor x, int64_t idx) {
  return (x - x[idx]).pow_(2).sum(1);
}

torch::Tensor fps_cpu(torch::Tensor src, torch::Tensor ptr, torch::Tensor ratio,
                      bool random_start) {

  CHECK_CPU(src);
  CHECK_CPU(ptr);
  CHECK_CPU(ratio);
  CHECK_INPUT(ptr.dim() == 1);

  src = src.view({src.size(0), -1}).contiguous();
  ptr = ptr.contiguous();
  auto batch_size = ptr.numel() - 1;

  auto deg = ptr.narrow(0, 1, batch_size) - ptr.narrow(0, 0, batch_size);
  auto out_ptr = deg.toType(torch::kFloat) * ratio;
  out_ptr = out_ptr.ceil().toType(torch::kLong).cumsum(0);

  auto out = torch::empty(out_ptr[-1].data_ptr<int64_t>()[0], ptr.options());

  auto ptr_data = ptr.data_ptr<int64_t>();
  auto out_ptr_data = out_ptr.data_ptr<int64_t>();
  auto out_data = out.data_ptr<int64_t>();

  int64_t grain_size = 1; // Always parallelize over batch dimension.
  at::parallel_for(0, batch_size, grain_size, [&](int64_t begin, int64_t end) {
    int64_t src_start, src_end, out_start, out_end;
    for (int64_t b = begin; b < end; b++) {
      src_start = ptr_data[b], src_end = ptr_data[b + 1];
      out_start = b == 0 ? 0 : out_ptr_data[b - 1], out_end = out_ptr_data[b];

      auto y = src.narrow(0, src_start, src_end - src_start);

      int64_t start_idx = 0;
      if (random_start)
        start_idx = rand() % y.size(0);

      out_data[out_start] = src_start + start_idx;
      auto dist = get_dist(y, start_idx);

      for (int64_t i = 1; i < out_end - out_start; i++) {
        int64_t argmax = dist.argmax().data_ptr<int64_t>()[0];
        out_data[out_start + i] = src_start + argmax;
        dist = torch::min(dist, get_dist(y, argmax));
      }
    }
  });

  return out;
}
