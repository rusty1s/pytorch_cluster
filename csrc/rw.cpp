#ifdef WITH_PYTHON
#include <Python.h>
#endif
#include <torch/script.h>

#include "cpu/rw_cpu.h"

#ifdef WITH_CUDA
#include "cuda/rw_cuda.h"
#endif

#ifdef _WIN32
#ifdef WITH_PYTHON
#ifdef WITH_CUDA
PyMODINIT_FUNC PyInit__rw_cuda(void) { return NULL; }
#else
PyMODINIT_FUNC PyInit__rw_cpu(void) { return NULL; }
#endif
#endif
#endif

CLUSTER_API std::tuple<torch::Tensor, torch::Tensor>
random_walk(torch::Tensor rowptr, torch::Tensor col, torch::Tensor start,
            int64_t walk_length, double p, double q) {
  if (rowptr.device().is_cuda()) {
#ifdef WITH_CUDA
    return random_walk_cuda(rowptr, col, start, walk_length, p, q);
#else
    AT_ERROR("Not compiled with CUDA support");
#endif
  } else {
    return random_walk_cpu(rowptr, col, start, walk_length, p, q);
  }
}

static auto registry =
    torch::RegisterOperators().op("torch_cluster::random_walk", &random_walk);
