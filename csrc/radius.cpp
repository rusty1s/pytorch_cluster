#include <Python.h>
#include <torch/script.h>

#ifdef WITH_CUDA
#include "cuda/radius_cuda.h"
#endif
#include "cpu/radius_cpu.h"

#ifdef _WIN32
PyMODINIT_FUNC PyInit__radius(void) { return NULL; }
#endif

torch::Tensor radius(torch::Tensor x, torch::Tensor y, torch::Tensor ptr_x,
                     torch::Tensor ptr_y, double r, int64_t max_num_neighbors) {
  if (x.device().is_cuda()) {
#ifdef WITH_CUDA
    return radius_cuda(x, y, ptr_x, ptr_y, r, max_num_neighbors);
#else
    AT_ERROR("Not compiled with CUDA support");
#endif
  } else {
    return radius_cpu(x, y, ptr_x, ptr_y, r, max_num_neighbors);
  }
}

static auto registry =
    torch::RegisterOperators().op("torch_cluster::radius", &radius);
