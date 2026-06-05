#pragma once
#include <torch/library.h>

#include "extensions.h"

namespace cluster {
CLUSTER_API int64_t cuda_version() noexcept;

namespace detail {
CLUSTER_INLINE_VARIABLE int64_t _cuda_version = cuda_version();
} // namespace detail
} // namespace cluster

CLUSTER_API torch::Tensor fps(torch::Tensor src, torch::Tensor ptr, torch::Tensor ratio,
                  bool random_start);

CLUSTER_API torch::Tensor graclus(torch::Tensor rowptr, torch::Tensor col,
                      std::optional<torch::Tensor> optional_weight);

CLUSTER_API torch::Tensor grid(torch::Tensor pos, torch::Tensor size,
                   std::optional<torch::Tensor> optional_start,
                   std::optional<torch::Tensor> optional_end);

CLUSTER_API torch::Tensor knn(torch::Tensor x, torch::Tensor y,
                  std::optional<torch::Tensor> ptr_x,
                  std::optional<torch::Tensor> ptr_y, int64_t k, bool cosine,
                  int64_t num_workers);

CLUSTER_API torch::Tensor nearest(torch::Tensor x, torch::Tensor y, torch::Tensor ptr_x,
                      torch::Tensor ptr_y);

CLUSTER_API torch::Tensor radius(torch::Tensor x, torch::Tensor y, torch::Tensor ptr_x,
                     torch::Tensor ptr_y, double r, int64_t max_num_neighbors);

CLUSTER_API std::tuple<torch::Tensor, torch::Tensor>
random_walk(torch::Tensor rowptr, torch::Tensor col, torch::Tensor start,
            int64_t walk_length, double p, double q);

CLUSTER_API torch::Tensor neighbor_sampler(torch::Tensor start, torch::Tensor rowptr,
                               int64_t count, double factor);

TORCH_LIBRARY(torch_cluster, m) {
    m.def("fps(Tensor src, Tensor ptr, Tensor ratio, bool random_start = False) -> Tensor");
    m.def("graclus(Tensor rowptr, Tensor col, Tensor? weight = None) -> Tensor");
    m.def("grid(Tensor pos, Tensor size, Tensor? start = None, Tensor? end = None) -> Tensor");
    m.def("knn(Tensor a, Tensor b, Tensor? ptr_x, Tensor? ptr_y, int k, bool cosine = False, int num_workers = 1) -> Tensor");
    m.def("nearest(Tensor x, Tensor y, Tensor ptr_x, Tensor ptr_y) -> Tensor");
    m.def("radius(Tensor x, Tensor y, Tensor? ptr_x, Tensor? ptr_y, float r, int max_num_neighbors, int num_workers = 1, bool ignore_same_index = False) -> Tensor");
    m.def("random_walk(Tensor rowptr, Tensor col, Tensor start, int walk_length, float p = 1, float q = 1) -> (Tensor, Tensor)");
    m.def("neighbor_sampler(Tensor start, Tensor rowptr, int count, float factor) -> Tensor");
    m.def("cuda_version() -> int");
}
