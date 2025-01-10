#pragma once

#include "../extensions.h"
torch::Tensor grid_cpu(torch::Tensor pos, torch::Tensor size,
                       std::optional<torch::Tensor> optional_start,
                       std::optional<torch::Tensor> optional_end);
