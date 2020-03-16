#pragma once

#include <torch/extension.h>

torch::Tensor grid_cpu(torch::Tensor pos, torch::Tensor size,
                       torch::optional<torch::Tensor> optional_start,
                       torch::optional<torch::Tensor> optional_end);
