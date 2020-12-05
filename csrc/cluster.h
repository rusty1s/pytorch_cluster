#pragma once

#include <torch/extension.h>

int64_t cuda_version();

torch::Tensor fps(torch::Tensor src, torch::Tensor ptr, double ratio,
                  bool random_start);

torch::Tensor graclus(torch::Tensor rowptr, torch::Tensor col,
                      torch::optional<torch::Tensor> optional_weight);

torch::Tensor grid(torch::Tensor pos, torch::Tensor size,
                   torch::optional<torch::Tensor> optional_start,
                   torch::optional<torch::Tensor> optional_end);

torch::Tensor knn(torch::Tensor x, torch::Tensor y, torch::Tensor ptr_x,
                  torch::Tensor ptr_y, int64_t k, bool cosine);

torch::Tensor nearest(torch::Tensor x, torch::Tensor y, torch::Tensor ptr_x,
                      torch::Tensor ptr_y);

torch::Tensor radius(torch::Tensor x, torch::Tensor y, torch::Tensor ptr_x,
                     torch::Tensor ptr_y, double r, int64_t max_num_neighbors);

std::tuple<torch::Tensor, torch::Tensor>
random_walk(torch::Tensor rowptr, torch::Tensor col, torch::Tensor start,
            int64_t walk_length, double p, double q);

torch::Tensor neighbor_sampler(torch::Tensor start, torch::Tensor rowptr,
                               int64_t count, double factor);
