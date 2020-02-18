#include <Python.h>
#include <torch/script.h>

#include "cpu/graclus_cpu.h"

#ifdef WITH_CUDA
#include "cuda/graclus_cuda.h"
#endif

#ifdef _WIN32
PyMODINIT_FUNC PyInit__graclus(void) { return NULL; }
#endif

torch::Tensor graclus(torch::Tensor row, torch::Tensor col,
                      torch::optional<torch::Tensor> optional_weight,
                      int64_t num_nodes) {
  if (row.device().is_cuda()) {
#ifdef WITH_CUDA
    return graclus_cuda(row, col, optional_weight, num_nodes);
#else
    AT_ERROR("Not compiled with CUDA support");
#endif
  } else {
    return graclus_cpu(row, col, optional_weight, num_nodes);
  }
}

static auto registry =
    torch::RegisterOperators().op("torch_cluster::graclus", &graclus);
