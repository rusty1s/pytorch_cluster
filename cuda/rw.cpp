#include <torch/extension.h>

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be CUDA tensor")
#define IS_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " is not contiguous");

at::Tensor rw_cuda(at::Tensor row, at::Tensor col, at::Tensor start,
                   size_t walk_length, float p, float q, size_t num_nodes);

at::Tensor rw(at::Tensor row, at::Tensor col, at::Tensor start,
              size_t walk_length, float p, float q, size_t num_nodes) {
  CHECK_CUDA(row);
  CHECK_CUDA(col);
  CHECK_CUDA(start);
  return rw_cuda(row, col, start, walk_length, p, q, num_nodes);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rw", &rw, "Random Walk Sampling (CUDA)");
}
