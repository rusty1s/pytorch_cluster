#pragma once

#include "../extensions.h"

std::tuple<torch::Tensor, torch::Tensor>
random_walk_cuda(torch::Tensor rowptr, torch::Tensor col, torch::Tensor start,
                 int64_t walk_length, double p, double q);

std::tuple<torch::Tensor, torch::Tensor>
random_walk_weighted_cuda(torch::Tensor rowptr, torch::Tensor col,
                          torch::Tensor edge_weight, torch::Tensor start,
                          int64_t walk_length, double p, double q);