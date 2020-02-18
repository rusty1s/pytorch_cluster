#pragma once

#include <torch/extension.h>

torch::Tensor graclus_cpu(torch::Tensor row, torch::Tensor col,
                          torch::optional<torch::Tensor> optional_weight,
                          int64_t num_nodes);
