#include <torch/extension.h>

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be CUDA tensor")

at::Tensor graclus_cuda(at::Tensor row, at::Tensor col, int64_t num_nodes);

at::Tensor weighted_graclus_cuda(at::Tensor row, at::Tensor col,
                                 at::Tensor weight, int64_t num_nodes);

at::Tensor graclus(at::Tensor row, at::Tensor col, int64_t num_nodes) {
  CHECK_CUDA(row);
  CHECK_CUDA(col);
  return graclus_cuda(row, col, num_nodes);
}

at::Tensor weighted_graclus(at::Tensor row, at::Tensor col, at::Tensor weight,
                            int64_t num_nodes) {
  CHECK_CUDA(row);
  CHECK_CUDA(col);
  CHECK_CUDA(weight);
  return graclus_cuda(row, col, num_nodes);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("graclus", &graclus, "Graclus (CUDA)");
  m.def("weighted_graclus", &weighted_graclus, "Weighted Graclus (CUDA)");
}
