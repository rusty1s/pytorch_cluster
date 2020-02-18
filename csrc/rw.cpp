#include <Python.h>
#include <torch/script.h>

#include "cpu/rw_cpu.h"

#ifdef WITH_CUDA
#include "cuda/rw_cuda.h"
#endif

#ifdef _WIN32
PyMODINIT_FUNC PyInit__grid(void) { return NULL; }
#endif

torch::Tensor grid(torch::Tensor pos, torch::Tensor size,
                   torch::optional<torch::Tensor> optional_start,
                   torch::optional<torch::Tensor> optional_end) {
  if (pos.device().is_cuda()) {
#ifdef WITH_CUDA
    AT_ERROR("No CUDA version supported.")
#else
    AT_ERROR("Not compiled with CUDA support");
#endif
  } else {
    return grid_cpu(pos, size, optional_start, optional_end);
  }
}

static auto registry =
    torch::RegisterOperators().op("torch_cluster::grid", &grid);
