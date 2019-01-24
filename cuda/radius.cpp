#include <torch/extension.h>

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be CUDA tensor")
#define IS_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " is not contiguous");

at::Tensor radius_cuda(at::Tensor x, at::Tensor y, float radius,
                       at::Tensor batch_x, at::Tensor batch_y,
                       size_t max_num_neighbors);

at::Tensor radius(at::Tensor x, at::Tensor y, float radius, at::Tensor batch_x,
                  at::Tensor batch_y, size_t max_num_neighbors) {
  CHECK_CUDA(x);
  IS_CONTIGUOUS(x);
  CHECK_CUDA(y);
  IS_CONTIGUOUS(y);
  CHECK_CUDA(batch_x);
  CHECK_CUDA(batch_y);
  return radius_cuda(x, y, radius, batch_x, batch_y, max_num_neighbors);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("radius", &radius, "Radius (CUDA)");
}
