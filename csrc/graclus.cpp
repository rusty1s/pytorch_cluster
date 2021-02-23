#include <Python.h>
#include <torch/script.h>

#include "cpu/graclus_cpu.h"

#ifdef WITH_CUDA
#include "cuda/graclus_cuda.h"
#endif

#ifdef _WIN32
#ifdef WITH_CUDA
PyMODINIT_FUNC PyInit__graclus_cuda(void) { return NULL; }
#else
PyMODINIT_FUNC PyInit__graclus_cpu(void) { return NULL; }
#endif
#endif

torch::Tensor graclus(torch::Tensor rowptr, torch::Tensor col,
                      torch::optional<torch::Tensor> optional_weight) {
  if (rowptr.device().is_cuda()) {
#ifdef WITH_CUDA
    return graclus_cuda(rowptr, col, optional_weight);
#else
    AT_ERROR("Not compiled with CUDA support");
#endif
  } else {
    return graclus_cpu(rowptr, col, optional_weight);
  }
}

static auto registry =
    torch::RegisterOperators().op("torch_cluster::graclus", &graclus);
