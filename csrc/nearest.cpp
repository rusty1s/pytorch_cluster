#ifdef WITH_PYTHON
#include <Python.h>
#endif
#include <torch/torch.h>
#include <torch/library.h>

#include "extensions.h"

#ifdef WITH_CUDA
#include "cuda/nearest_cuda.h"
#endif

#ifdef _WIN32
#ifdef WITH_PYTHON
#ifdef WITH_CUDA
PyMODINIT_FUNC PyInit__nearest_cuda(void) { return NULL; }
#else
PyMODINIT_FUNC PyInit__nearest_cpu(void) { return NULL; }
#endif
#endif
#endif

CLUSTER_API torch::Tensor nearest(torch::Tensor x, torch::Tensor y, torch::Tensor ptr_x,
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

#ifdef WITH_CUDA
TORCH_LIBRARY_IMPL(torch_cluster, CUDA, m) {
    m.impl("nearest", &nearest_cuda);
}
TORCH_LIBRARY_IMPL(torch_cluster, HIP, m) {
  m.impl("nearest", &nearest_cuda);
}
#endif
