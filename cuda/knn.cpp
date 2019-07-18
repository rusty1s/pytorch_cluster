#include <torch/extension.h>

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be CUDA tensor")
#define IS_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " is not contiguous");

at::Tensor knn_cuda(at::Tensor x, at::Tensor y, size_t k, at::Tensor batch_x,
                    at::Tensor batch_y, bool cosine);

at::Tensor knn(at::Tensor x, at::Tensor y, size_t k, at::Tensor batch_x,
               at::Tensor batch_y, bool cosine) {
  CHECK_CUDA(x);
  IS_CONTIGUOUS(x);
  CHECK_CUDA(y);
  IS_CONTIGUOUS(y);
  CHECK_CUDA(batch_x);
  CHECK_CUDA(batch_y);
  return knn_cuda(x, y, k, batch_x, batch_y, cosine);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("knn", &knn, "k-Nearest Neighbor (CUDA)");
}
