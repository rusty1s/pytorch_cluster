#pragma once

#include <torch/extension.h>

torch::Tensor neighbor_sampler_cpu(torch::Tensor start, torch::Tensor rowptr,
                                   int64_t count, double factor);
