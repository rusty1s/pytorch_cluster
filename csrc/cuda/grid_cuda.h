#pragma once

#include "../extensions.h"

torch::Tensor grid_cuda(torch::Tensor pos, torch::Tensor size,
                        std::optional<torch::Tensor> optional_start,
                        std::optional<torch::Tensor> optional_end);
