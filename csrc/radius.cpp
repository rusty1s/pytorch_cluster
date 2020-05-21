#include <Python.h>
#include <torch/script.h>

#ifdef WITH_CUDA
#include "cuda/radius_cuda.h"
#endif
#include "cpu/radius_cpu.h"

#ifdef _WIN32
PyMODINIT_FUNC PyInit__radius(void) { return NULL; }
#endif

torch::Tensor radius(torch::Tensor x, torch::Tensor y, torch::optional<torch::Tensor> ptr_x,
                     torch::optional<torch::Tensor> ptr_y, double r, int64_t max_num_neighbors) {
  if (x.device().is_cuda()) {
#ifdef WITH_CUDA
    if (!(ptr_x.has_value()) && !(ptr_y.has_value())) {
      auto batch_x = torch::tensor({0,torch::size(x,0)}).to(torch::kLong).to(torch::kCUDA);
      auto batch_y = torch::tensor({0,torch::size(y,0)}).to(torch::kLong).to(torch::kCUDA);
      return radius_cuda(x, y, batch_x, batch_y, r, max_num_neighbors);
    }
    else if (!(ptr_x.has_value())) {
      auto batch_x = torch::tensor({0,torch::size(x,0)}).to(torch::kLong).to(torch::kCUDA);
      auto batch_y = ptr_y.value();
      return radius_cuda(x, y, batch_x, batch_y, r, max_num_neighbors);
    }
    else if (!(ptr_y.has_value())) {
      auto batch_x = ptr_x.value();
      auto batch_y = torch::tensor({0,torch::size(y,0)}).to(torch::kLong).to(torch::kCUDA);
      return radius_cuda(x, y, batch_x, batch_y, r, max_num_neighbors);
    }
    auto batch_x = ptr_x.value();
    auto batch_y = ptr_y.value();
    return radius_cuda(x, y, batch_x, batch_y, r, max_num_neighbors);
#else
    AT_ERROR("Not compiled with CUDA support");
#endif
  } else {
    if (!(ptr_x.has_value()) && !(ptr_y.has_value())) {
      return radius_cpu(x,y,r,max_num_neighbors);
    }
    if (!(ptr_x.has_value())) {
      auto batch_x = torch::zeros({torch::size(x,0)}).to(torch::kLong);
      auto batch_y = ptr_y.value();
      return batch_radius_cpu(x, y, batch_x, batch_y, r, max_num_neighbors);
    }
    else if (!(ptr_y.has_value())) {
      auto batch_x = ptr_x.value();
      auto batch_y = torch::zeros({torch::size(y,0)}).to(torch::kLong);
      return batch_radius_cpu(x, y, batch_x, batch_y, r, max_num_neighbors);
    }
    auto batch_x = ptr_x.value();
    auto batch_y = ptr_y.value();
    return batch_radius_cpu(x, y, batch_x, batch_y, r, max_num_neighbors);
  }
}

static auto registry =
    torch::RegisterOperators().op("torch_cluster::radius", &radius);
