#include <Python.h>
#include <torch/script.h>

#include "cpu/fps_cpu.h"

#ifdef WITH_CUDA
#include "cuda/fps_cuda.h"
#endif

#ifdef _WIN32
#ifdef WITH_CUDA
PyMODINIT_FUNC PyInit__fps_cuda(void) { return NULL; }
#else
PyMODINIT_FUNC PyInit__fps_cpu(void) { return NULL; }
#endif
#endif

torch::Tensor fps(torch::Tensor src, torch::Tensor ptr, torch::Tensor ratio,
                  bool random_start) {
  if (src.device().is_cuda()) {
#ifdef WITH_CUDA
    return fps_cuda(src, ptr, ratio, random_start);
#else
    AT_ERROR("Not compiled with CUDA support");
#endif
  } else {
    return fps_cpu(src, ptr, ratio, random_start);
  }
}

static auto registry =
    torch::RegisterOperators().op("torch_cluster::fps", &fps);
