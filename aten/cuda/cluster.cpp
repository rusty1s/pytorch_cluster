#include <torch/torch.h>

at::Tensor grid(at::Tensor pos, at::Tensor size, at::Tensor start,
                at::Tensor end);

at::Tensor graclus(at::Tensor row, at::Tensor col, int num_nodes);

at::Tensor weighted_graclus(at::Tensor row, at::Tensor col, at::Tensor weight,
                            int num_nodes);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("grid", &grid, "Grid (CUDA)");
  m.def("graclus", &graclus, "Graclus (CUDA)");
  m.def("weighted_graclus", &weighted_graclus, "Weightes Graclus (CUDA)");
}
