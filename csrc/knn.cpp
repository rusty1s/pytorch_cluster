#ifdef WITH_PYTHON
#include <Python.h>
#endif
#include <torch/script.h>

#include "cpu/knn_cpu.h"

#ifdef WITH_CUDA
#include "cuda/knn_cuda.h"
#endif

#ifdef _WIN32
#ifdef WITH_PYTHON
#ifdef WITH_CUDA
PyMODINIT_FUNC PyInit__knn_cuda(void) { return NULL; }
#else
PyMODINIT_FUNC PyInit__knn_cpu(void) { return NULL; }
#endif
#endif
#endif

CLUSTER_API torch::Tensor knn(torch::Tensor x, torch::Tensor y,
                  torch::optional<torch::Tensor> ptr_x,
                  torch::optional<torch::Tensor> ptr_y, int64_t k, bool cosine,
                  int64_t num_workers) {
  if (x.device().is_cuda()) {
#ifdef WITH_CUDA
    return knn_cuda(x, y, ptr_x, ptr_y, k, cosine);
#else
    AT_ERROR("Not compiled with CUDA support");
#endif
  } else {
    if (cosine)
      AT_ERROR("`cosine` argument not supported on CPU");
    return knn_cpu(x, y, ptr_x, ptr_y, k, num_workers);
  }
}

static auto registry =
    torch::RegisterOperators().op("torch_cluster::knn", &knn);
