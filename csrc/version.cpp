#ifdef WITH_PYTHON
#include <Python.h>
#endif
#include <torch/script.h>
#include "cluster.h"
#include "macros.h"

#ifdef WITH_CUDA
#include <cuda.h>
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
  return CUDA_VERSION;
#else
  return -1;
#endif
}
} // namespace sparse


static auto registry =
    torch::RegisterOperators().op("torch_cluster::cuda_version", &cluster::cuda_version);
