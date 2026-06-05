#ifdef WITH_PYTHON
#include <Python.h>
#endif
#include <torch/torch.h>
#include <torch/library.h>

#include "cpu/sampler_cpu.h"

#ifdef _WIN32
#ifdef WITH_PYTHON
#ifdef WITH_CUDA
PyMODINIT_FUNC PyInit__sampler_cuda(void) { return NULL; }
#else
PyMODINIT_FUNC PyInit__sampler_cpu(void) { return NULL; }
#endif
#endif
#endif

CLUSTER_API torch::Tensor neighbor_sampler(torch::Tensor start, torch::Tensor rowptr,
                               int64_t count, double factor) {
  if (rowptr.device().is_cuda()) {
#ifdef WITH_CUDA
    AT_ERROR("No CUDA version supported");
#else
    AT_ERROR("Not compiled with CUDA support");
#endif
  } else {
    return neighbor_sampler_cpu(start, rowptr, count, factor);
  }
}

TORCH_LIBRARY_IMPL(torch_cluster, CPU, m) {
  m.impl("neighbor_sampler", &neighbor_sampler_cpu);
}
