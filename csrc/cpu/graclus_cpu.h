#pragma once

#include <torch/extension.h>

torch::Tensor graclus_cpu(torch::Tensor rowptr, torch::Tensor col,
                          torch::optional<torch::Tensor> optional_weight);
