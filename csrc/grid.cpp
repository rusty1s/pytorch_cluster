#ifdef WITH_PYTHON
#include <Python.h>
#endif
#include <torch/torch.h>
#include <torch/library.h>

#include "cpu/grid_cpu.h"

#ifdef WITH_CUDA
#include "cuda/grid_cuda.h"
#endif

#ifdef _WIN32
#ifdef WITH_PYTHON
#ifdef WITH_CUDA
PyMODINIT_FUNC PyInit__grid_cuda(void) { return NULL; }
#else
PyMODINIT_FUNC PyInit__grid_cpu(void) { return NULL; }
#endif
#endif
#endif

CLUSTER_API torch::Tensor grid(torch::Tensor pos, torch::Tensor size,
                   std::optional<torch::Tensor> optional_start,
                   std::optional<torch::Tensor> optional_end) {
  if (pos.device().is_cuda()) {
#ifdef WITH_CUDA
    return grid_cuda(pos, size, optional_start, optional_end);
#else
    AT_ERROR("Not compiled with CUDA support");
#endif
  } else {
    return grid_cpu(pos, size, optional_start, optional_end);
  }
}

TORCH_LIBRARY_IMPL(torch_cluster, CPU, m) {
    m.impl("grid", &grid_cpu);
}

#ifdef WITH_CUDA
TORCH_LIBRARY_IMPL(torch_cluster, CUDA, m) {
    m.impl("grid", &grid_cuda);
}
TORCH_LIBRARY_IMPL(torch_cluster, HIP, m) {
    m.impl("grid", &grid_cuda);
}
#endif
