#pragma once

#include <torch/extension.h>

torch::Tensor graclus_cuda(torch::Tensor rowptr, torch::Tensor col,
                           torch::optional<torch::Tensor> optional_weight);
