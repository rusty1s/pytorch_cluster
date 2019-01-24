#include <torch/extension.h>

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be CUDA tensor")
#define IS_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " is not contiguous");

at::Tensor nearest_cuda(at::Tensor x, at::Tensor y, at::Tensor batch_x,
                        at::Tensor batch_y);

at::Tensor nearest(at::Tensor x, at::Tensor y, at::Tensor batch_x,
                   at::Tensor batch_y) {
  CHECK_CUDA(x);
  IS_CONTIGUOUS(x);
  CHECK_CUDA(y);
  IS_CONTIGUOUS(y);
  CHECK_CUDA(batch_x);
  CHECK_CUDA(batch_y);
  return nearest_cuda(x, y, batch_x, batch_y);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("nearest", &nearest, "Nearest Neighbor (CUDA)");
}
