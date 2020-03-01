#include "fps_cuda.h"

#include "utils.cuh"

inline torch::Tensor get_dist(torch::Tensor x, int64_t idx) {
  return (x - x[idx]).norm(2, 1);
}

torch::Tensor fps_cuda(torch::Tensor src, torch::Tensor ptr, double ratio,
                       bool random_start) {

  CHECK_CUDA(src);
  CHECK_CUDA(ptr);
  CHECK_INPUT(ptr.dim() == 1);
  AT_ASSERTM(ratio > 0 and ratio < 1, "Invalid input");

  src = src.view({src.size(0), -1}).contiguous();
  ptr = ptr.contiguous();
  auto batch_size = ptr.size(0) - 1;

  auto deg = ptr.narrow(0, 1, batch_size) - ptr.narrow(0, 0, batch_size);
  auto out_ptr = deg.toType(torch::kFloat) * (float)ratio;
  out_ptr = out_ptr.ceil().toType(torch::kLong).cumsum(0);
  out_ptr = torch::cat({torch.zeros(1, ptr.options()), out_ptr}, 0);

  torch::Tensor start;
  if (random_start) {
    start = at::rand(batch_size, src.options());
    start = (start * deg.toType(torch::kFloat)).toType(torch::kLong);
  } else {
    start = torch::zeros(batch_size, ptr.options());
  }

  auto out = torch::empty(out_ptr[-1].data_ptr<int64_t>()[0], ptr.options());

  auto ptr_data = ptr.data_ptr<int64_t>();
  auto out_ptr_data = out_ptr.data_ptr<int64_t>();
  auto out_data = out.data_ptr<int64_t>();

  return out;
}
