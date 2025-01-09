#pragma once

#include "../extensions.h"

torch::Tensor graclus_cuda(torch::Tensor rowptr, torch::Tensor col,
                           std::optional<torch::Tensor> optional_weight);
