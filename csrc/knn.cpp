#include <Python.h>
#include <torch/script.h>

#ifdef WITH_CUDA
#include "cuda/knn_cuda.h"
#endif

#ifdef _WIN32
PyMODINIT_FUNC PyInit__knn(void) { return NULL; }
#endif

torch::Tensor knn(torch::Tensor x, torch::Tensor y, torch::Tensor ptr_x,
                  torch::Tensor ptr_y, int64_t k, bool cosine) {
  if (x.device().is_cuda()) {
#ifdef WITH_CUDA
    return knn_cuda(x, y, ptr_x, ptr_y, k, cosine);
#else
    AT_ERROR("Not compiled with CUDA support");
#endif
  } else {
    AT_ERROR("No CPU version supported");
  }
}

static auto registry =
    torch::RegisterOperators().op("torch_cluster::knn", &knn);
