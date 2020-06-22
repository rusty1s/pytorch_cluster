#pragma once

#include <torch/extension.h>

torch::Tensor radius_cuda(torch::Tensor x, torch::Tensor y,
                          torch::optional<torch::Tensor> ptr_x,
                          torch::optional<torch::Tensor> ptr_y, double r,
                          int64_t max_num_neighbors);
