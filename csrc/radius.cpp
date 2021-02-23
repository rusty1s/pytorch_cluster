#include <Python.h>
#include <torch/script.h>

#include "cpu/radius_cpu.h"

#ifdef WITH_CUDA
#include "cuda/radius_cuda.h"
#endif

#ifdef _WIN32
#ifdef WITH_CUDA
PyMODINIT_FUNC PyInit__radius_cuda(void) { return NULL; }
#else
PyMODINIT_FUNC PyInit__radius_cpu(void) { return NULL; }
#endif
#endif

torch::Tensor radius(torch::Tensor x, torch::Tensor y,
                     torch::optional<torch::Tensor> ptr_x,
                     torch::optional<torch::Tensor> ptr_y, double r,
                     int64_t max_num_neighbors, int64_t num_workers) {
  if (x.device().is_cuda()) {
#ifdef WITH_CUDA
    return radius_cuda(x, y, ptr_x, ptr_y, r, max_num_neighbors);
#else
    AT_ERROR("Not compiled with CUDA support");
#endif
  } else {
    return radius_cpu(x, y, ptr_x, ptr_y, r, max_num_neighbors, num_workers);
  }
}

static auto registry =
    torch::RegisterOperators().op("torch_cluster::radius", &radius);
