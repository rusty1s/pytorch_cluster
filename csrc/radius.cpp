#ifdef WITH_PYTHON
#include <Python.h>
#endif
#include <torch/torch.h>
#include <torch/library.h>

#include "cpu/radius_cpu.h"

#ifdef WITH_CUDA
#include "cuda/radius_cuda.h"
#endif

#ifdef _WIN32
#ifdef WITH_PYTHON
#ifdef WITH_CUDA
PyMODINIT_FUNC PyInit__radius_cuda(void) { return NULL; }
#else
PyMODINIT_FUNC PyInit__radius_cpu(void) { return NULL; }
#endif
#endif
#endif

CLUSTER_API torch::Tensor radius(torch::Tensor x, torch::Tensor y,
                     std::optional<torch::Tensor> ptr_x,
                     std::optional<torch::Tensor> ptr_y, double r,
                     int64_t max_num_neighbors, int64_t num_workers,
                     bool ignore_same_index) {
  if (x.device().is_cuda()) {
#ifdef WITH_CUDA
    return radius_cuda(x, y, ptr_x, ptr_y, r, max_num_neighbors, ignore_same_index);
#else
    AT_ERROR("Not compiled with CUDA support");
#endif
  } else {
    return radius_cpu(x, y, ptr_x, ptr_y, r, max_num_neighbors, num_workers, ignore_same_index);
  }
}

TORCH_LIBRARY_IMPL(torch_cluster, CPU, m) {
  m.impl("radius", &radius_cpu);
}

#ifdef WITH_CUDA
inline torch::Tensor radius_cuda_wrap(torch::Tensor x, torch::Tensor y,
                     std::optional<torch::Tensor> ptr_x,
                     std::optional<torch::Tensor> ptr_y, double r,
                     int64_t max_num_neighbors, int64_t num_workers,
                     bool ignore_same_index) {
    return radius_cuda(x, y, ptr_x, ptr_y, r, max_num_neighbors, ignore_same_index);
}

TORCH_LIBRARY_IMPL(torch_cluster, CUDA, m) {
  m.impl("radius", &radius_cuda_wrap);
}
TORCH_LIBRARY_IMPL(torch_cluster, HIP, m) {
  m.impl("radius", &radius_cuda_wrap);
}
#endif
