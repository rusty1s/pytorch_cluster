#include <torch/extension.h>

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be CUDA tensor")
#define IS_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " is not contiguous");

at::Tensor fps_cuda(at::Tensor x, at::Tensor batch, float ratio, bool random);

at::Tensor fps(at::Tensor x, at::Tensor batch, float ratio, bool random) {
  CHECK_CUDA(x);
  IS_CONTIGUOUS(x);
  CHECK_CUDA(batch);
  return fps_cuda(x, batch, ratio, random);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fps", &fps, "Farthest Point Sampling (CUDA)");
}
