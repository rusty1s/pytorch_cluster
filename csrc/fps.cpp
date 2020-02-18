#include <Python.h>
#include <torch/script.h>

#include "cpu/fps_cpu.h"

#ifdef WITH_CUDA
#include "cuda/fps_cuda.h"
#endif

#ifdef _WIN32
PyMODINIT_FUNC PyInit__fps(void) { return NULL; }
#endif

torch::Tensor fps(torch::Tensor src,
                  torch::optional<torch::Tensor> optional_ptr, double ratio,
                  bool random_start) {
  if (src.device().is_cuda()) {
#ifdef WITH_CUDA
    return fps_cuda(src, optional_ptr, ratio, random_start);
#else
    AT_ERROR("Not compiled with CUDA support");
#endif
  } else {
    return fps_cpu(src, optional_ptr, ratio, random_start);
  }
}

static auto registry =
    torch::RegisterOperators().op("torch_cluster::fps", &fps);
