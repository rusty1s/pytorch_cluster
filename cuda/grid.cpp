#include <torch/extension.h>

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be CUDA tensor")

at::Tensor grid_cuda(at::Tensor pos, at::Tensor size, at::Tensor start,
                     at::Tensor end);

at::Tensor grid(at::Tensor pos, at::Tensor size, at::Tensor start,
                at::Tensor end) {
  CHECK_CUDA(pos);
  CHECK_CUDA(size);
  CHECK_CUDA(start);
  CHECK_CUDA(end);
  return grid_cuda(pos, size, start, end);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("grid", &grid, "Grid (CUDA)");
}
