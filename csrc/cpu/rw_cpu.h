#pragma once

#include <torch/extension.h>

at::Tensor random_walk_cpu(torch::Tensor row, torch::Tensor col,
                           torch::Tensor start, int64_t walk_length, double p,
                           double q, int64_t num_nodes);
