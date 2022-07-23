#pragma once

#include "extensions.h"

namespace cluster {
CLUSTER_API int64_t cuda_version() noexcept;

namespace detail {
CLUSTER_INLINE_VARIABLE int64_t _cuda_version = cuda_version();
} // namespace detail
} // namespace cluster

CLUSTER_API torch::Tensor fps(torch::Tensor src, torch::Tensor ptr, double ratio,
                  bool random_start);

CLUSTER_API torch::Tensor graclus(torch::Tensor rowptr, torch::Tensor col,
                      torch::optional<torch::Tensor> optional_weight);

CLUSTER_API torch::Tensor grid(torch::Tensor pos, torch::Tensor size,
                   torch::optional<torch::Tensor> optional_start,
                   torch::optional<torch::Tensor> optional_end);

CLUSTER_API torch::Tensor knn(torch::Tensor x, torch::Tensor y, torch::Tensor ptr_x,
                  torch::Tensor ptr_y, int64_t k, bool cosine);

CLUSTER_API torch::Tensor nearest(torch::Tensor x, torch::Tensor y, torch::Tensor ptr_x,
                      torch::Tensor ptr_y);

CLUSTER_API torch::Tensor radius(torch::Tensor x, torch::Tensor y, torch::Tensor ptr_x,
                     torch::Tensor ptr_y, double r, int64_t max_num_neighbors);

CLUSTER_API std::tuple<torch::Tensor, torch::Tensor>
random_walk(torch::Tensor rowptr, torch::Tensor col, torch::Tensor start,
            int64_t walk_length, double p, double q);

CLUSTER_API torch::Tensor neighbor_sampler(torch::Tensor start, torch::Tensor rowptr,
                               int64_t count, double factor);
