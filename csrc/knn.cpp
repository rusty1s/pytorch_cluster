#include <Python.h>
#include <torch/script.h>

#ifdef WITH_CUDA
#include "cuda/knn_cuda.h"
#endif
#include "cpu/knn_cpu.h"

#ifdef _WIN32
PyMODINIT_FUNC PyInit__knn(void) { return NULL; }
#endif

torch::Tensor knn(torch::Tensor x, torch::Tensor y, torch::optional<torch::Tensor> ptr_x,
                  torch::optional<torch::Tensor> ptr_y, int64_t k, bool cosine, int64_t n_threads) {
  if (x.device().is_cuda()) {
#ifdef WITH_CUDA
  if (!(ptr_x.has_value()) && !(ptr_y.has_value())) {
    auto batch_x = torch::tensor({0,torch::size(x,0)}).to(torch::kLong).to(torch::kCUDA);
    auto batch_y = torch::tensor({0,torch::size(y,0)}).to(torch::kLong).to(torch::kCUDA);
    return knn_cuda(x, y, batch_x, batch_y, k, cosine);
  }
  else if (!(ptr_x.has_value())) {
    auto batch_x = torch::tensor({0,torch::size(x,0)}).to(torch::kLong).to(torch::kCUDA);
    auto batch_y = ptr_y.value();
    return knn_cuda(x, y, batch_x, batch_y, k, cosine);
  }
  else if (!(ptr_y.has_value())) {
    auto batch_x = ptr_x.value();
    auto batch_y = torch::tensor({0,torch::size(y,0)}).to(torch::kLong).to(torch::kCUDA);
    return knn_cuda(x, y, batch_x, batch_y, k, cosine);
  }
  auto batch_x = ptr_x.value();
  auto batch_y = ptr_y.value();
  return knn_cuda(x, y, batch_x, batch_y, k, cosine);
#else
    AT_ERROR("Not compiled with CUDA support");
#endif
  } else {
    if (cosine) {
      AT_ERROR("`cosine` argument not supported on CPU");
    }
    if (!(ptr_x.has_value()) && !(ptr_y.has_value())) {
      return knn_cpu(x,y,k,n_threads);
    }
    if (!(ptr_x.has_value())) {
      auto batch_x = torch::zeros({torch::size(x,0)}).to(torch::kLong);
      auto batch_y = ptr_y.value();
      return batch_knn_cpu(x, y, batch_x, batch_y, k);
    }
    else if (!(ptr_y.has_value())) {
      auto batch_x = ptr_x.value();
      auto batch_y = torch::zeros({torch::size(y,0)}).to(torch::kLong);
      return batch_knn_cpu(x, y, batch_x, batch_y, k);
    }
    auto batch_x = ptr_x.value();
    auto batch_y = ptr_y.value();
    return batch_knn_cpu(x, y, batch_x, batch_y, k);
  }
}

static auto registry =
    torch::RegisterOperators().op("torch_cluster::knn", &knn);
