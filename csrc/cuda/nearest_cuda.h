#pragma once

#include <torch/extension.h>

torch::Tensor nearest_cuda(torch::Tensor x, torch::Tensor y,
                           torch::Tensor ptr_x, torch::Tensor ptr_y);
