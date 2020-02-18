#include <Python.h>
#include <torch/script.h>

#include "cpu/sampler_cpu.h"

#ifdef _WIN32
PyMODINIT_FUNC PyInit__sampler(void) { return NULL; }
#endif

torch::Tensor neighbor_sampler(torch::Tensor start, torch::Tensor rowptr,
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

static auto registry = torch::RegisterOperators().op(
    "torch_cluster::neighbor_sampler", &neighbor_sampler);
