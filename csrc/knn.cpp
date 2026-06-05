#ifdef WITH_PYTHON
#include <Python.h>
#endif
#include <torch/torch.h>
#include <torch/library.h>

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
                  std::optional<torch::Tensor> ptr_x,
                  std::optional<torch::Tensor> ptr_y, int64_t k, bool cosine,
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

inline torch::Tensor knn_cpu_wrap(torch::Tensor x, torch::Tensor y,
                  std::optional<torch::Tensor> ptr_x,
                  std::optional<torch::Tensor> ptr_y, int64_t k, bool cosine,
                  int64_t num_workers = 1) {
    TORCH_CHECK(!cosine, "`cosine` argument not supported on CPU");
    return knn_cpu(x, y, ptr_x, ptr_y, k, num_workers);
}

TORCH_LIBRARY_IMPL(torch_cluster, CPU, m) {
  m.impl("knn", &knn_cpu_wrap);
}

#ifdef WITH_CUDA
inline torch::Tensor knn_cuda_wrap(torch::Tensor x, torch::Tensor y,
                  std::optional<torch::Tensor> ptr_x,
                  std::optional<torch::Tensor> ptr_y, int64_t k, bool cosine,
                  int64_t num_workers) {
    return knn_cuda(x, y, ptr_x, ptr_y, k, cosine);
}
TORCH_LIBRARY_IMPL(torch_cluster, CUDA, m) {
    m.impl("knn", &knn_cuda_wrap);
}
TORCH_LIBRARY_IMPL(torch_cluster, HIP, m) {
    m.impl("knn", &knn_cuda_wrap);
}
#endif
