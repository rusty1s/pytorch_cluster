#ifdef WITH_PYTHON
#include <Python.h>
#endif
#include "cluster.h"
#include "macros.h"
#include <torch/script.h>

#ifdef WITH_CUDA
#ifdef USE_ROCM
#include <hip/hip_version.h>
#else
#include <cuda.h>
#endif
#endif

#ifdef _WIN32
#ifdef WITH_PYTHON
#ifdef WITH_CUDA
PyMODINIT_FUNC PyInit__version_cuda(void) { return NULL; }
#else
PyMODINIT_FUNC PyInit__version_cpu(void) { return NULL; }
#endif
#endif
#endif

namespace cluster {
CLUSTER_API int64_t cuda_version() noexcept {
#ifdef WITH_CUDA
#ifdef USE_ROCM
  return HIP_VERSION;
#else
  return CUDA_VERSION;
#endif
#else
  return -1;
#endif
}
} // namespace cluster

static auto registry = torch::RegisterOperators().op(
    "torch_cluster::cuda_version", [] { return cluster::cuda_version(); });
