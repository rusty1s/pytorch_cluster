#pragma once

#include "../extensions.h"

torch::Tensor knn_cpu(torch::Tensor x, torch::Tensor y,
                      std::optional<torch::Tensor> ptr_x,
                      std::optional<torch::Tensor> ptr_y, int64_t k,
                      int64_t num_workers);
