#pragma once

#include <torch/extension.h>

torch::Tensor fps_cpu(torch::Tensor src,
                      torch::optional<torch::Tensor> optional_ptr, double ratio,
                      bool random_start);
