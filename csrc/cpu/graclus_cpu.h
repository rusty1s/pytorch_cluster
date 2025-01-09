#pragma once

#include "../extensions.h"

torch::Tensor graclus_cpu(torch::Tensor rowptr, torch::Tensor col,
                          std::optional<torch::Tensor> optional_weight);
