#pragma once

#include <torch/extension.h>

torch::Tensor fps_cpu(torch::Tensor src, torch::Tensor ptr, torch::Tensor ratio,
                      bool random_start);
