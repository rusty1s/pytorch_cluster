#include <torch/torch.h>

#define CHECK_CUDA(x) AT_ASSERT(x.type().is_cuda(), #x " must be a CUDA tensor")

#include "graclus.cpp"
#include "grid.cpp"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("graclus", &graclus, "Graclus (CUDA)");
  m.def("grid", &grid, "Grid (CUDA)");
}
