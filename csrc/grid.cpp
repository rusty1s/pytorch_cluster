#include <Python.h>
#include <torch/script.h>

#include "cpu/grid_cpu.h"

#ifdef WITH_CUDA
#include "cuda/grid_cuda.h"
#endif

#ifdef _WIN32
#ifdef WITH_CUDA
PyMODINIT_FUNC PyInit__grid_cuda(void) { return NULL; }
#else
PyMODINIT_FUNC PyInit__grid_cpu(void) { return NULL; }
#endif
#endif

torch::Tensor grid(torch::Tensor pos, torch::Tensor size,
                   torch::optional<torch::Tensor> optional_start,
                   torch::optional<torch::Tensor> optional_end) {
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

static auto registry =
    torch::RegisterOperators().op("torch_cluster::grid", &grid);
