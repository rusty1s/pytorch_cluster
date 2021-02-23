#include <Python.h>
#include <torch/script.h>

#ifdef WITH_CUDA
#include "cuda/nearest_cuda.h"
#endif

#ifdef _WIN32
#ifdef WITH_CUDA
PyMODINIT_FUNC PyInit__nearest_cuda(void) { return NULL; }
#else
PyMODINIT_FUNC PyInit__nearest_cpu(void) { return NULL; }
#endif
#endif

torch::Tensor nearest(torch::Tensor x, torch::Tensor y, torch::Tensor ptr_x,
                      torch::Tensor ptr_y) {
  if (x.device().is_cuda()) {
#ifdef WITH_CUDA
    return nearest_cuda(x, y, ptr_x, ptr_y);
#else
    AT_ERROR("Not compiled with CUDA support");
#endif
  } else {
    AT_ERROR("No CPU version supported");
  }
}

static auto registry =
    torch::RegisterOperators().op("torch_cluster::nearest", &nearest);
