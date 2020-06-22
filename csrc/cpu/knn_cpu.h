#pragma once

#include <torch/extension.h>

torch::Tensor knn_cpu(torch::Tensor x, torch::Tensor y,
                      torch::optional<torch::Tensor> ptr_x,
                      torch::optional<torch::Tensor> ptr_y, int64_t k,
                      int64_t num_workers);
