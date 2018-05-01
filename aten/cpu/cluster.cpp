#include <torch/torch.h>

#include "graclus.cpp"
#include "grid.cpp"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("graclus", &graclus, "Graclus (CPU)");
  m.def("grid", &grid, "Grid (CPU)");
}
