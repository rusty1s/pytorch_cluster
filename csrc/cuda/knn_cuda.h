#pragma once

#include <torch/extension.h>

torch::Tensor knn_cuda(torch::Tensor x, torch::Tensor y,
                       torch::optional<torch::Tensor> ptr_x,
                       torch::optional<torch::Tensor> ptr_y, int64_t k,
                       bool cosine);
