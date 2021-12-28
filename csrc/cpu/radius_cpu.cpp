#include "radius_cpu.h"

#include "utils.h"
#include "utils/KDTreeVectorOfVectorsAdaptor.h"
#include "utils/nanoflann.hpp"

torch::Tensor radius_cpu(torch::Tensor x, torch::Tensor y,
                         torch::optional<torch::Tensor> ptr_x,
                         torch::optional<torch::Tensor> ptr_y, double r,
                         int64_t max_num_neighbors, int64_t num_workers) {

  CHECK_CPU(x);
  CHECK_INPUT(x.dim() == 2);
  CHECK_CPU(y);
  CHECK_INPUT(y.dim() == 2);

  if (ptr_x.has_value()) {
    CHECK_CPU(ptr_x.value());
    CHECK_INPUT(ptr_x.value().dim() == 1);
  }
  if (ptr_y.has_value()) {
    CHECK_CPU(ptr_y.value());
    CHECK_INPUT(ptr_y.value().dim() == 1);
  }

  std::vector<size_t> out_vec = std::vector<size_t>();

  AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, x.scalar_type(), "_", [&] {
    // See: nanoflann/examples/vector_of_vectors_example.cpp

    auto x_data = x.data_ptr<scalar_t>();
    auto y_data = y.data_ptr<scalar_t>();
    typedef std::vector<std::vector<scalar_t>> vec_t;
    nanoflann::SearchParams params;
    params.sorted = false;

    if (!ptr_x.has_value()) { // Single example.

      vec_t pts(x.size(0));
      for (int64_t i = 0; i < x.size(0); i++) {
        pts[i].resize(x.size(1));
        for (int64_t j = 0; j < x.size(1); j++) {
          pts[i][j] = x_data[i * x.size(1) + j];
        }
      }

      typedef KDTreeVectorOfVectorsAdaptor<vec_t, scalar_t> my_kd_tree_t;

      my_kd_tree_t mat_index(x.size(1), pts, 10);
      mat_index.index->buildIndex();

      for (int64_t i = 0; i < y.size(0); i++) {
        std::vector<std::pair<size_t, scalar_t>> ret_matches;
        size_t num_matches = mat_index.index->radiusSearch(
            y_data + i * y.size(1), r * r, ret_matches, params);

        for (size_t j = 0; j < std::min(num_matches, (size_t)max_num_neighbors);
             j++) {
          out_vec.push_back(ret_matches[j].first);
          out_vec.push_back(i);
        }
      }

    } else { // Batch-wise.

      auto ptr_x_data = ptr_x.value().data_ptr<int64_t>();
      auto ptr_y_data = ptr_y.value().data_ptr<int64_t>();

      for (int64_t b = 0; b < ptr_x.value().size(0) - 1; b++) {
        auto x_start = ptr_x_data[b], x_end = ptr_x_data[b + 1];
        auto y_start = ptr_y_data[b], y_end = ptr_y_data[b + 1];

        if (x_start == x_end || y_start == y_end)
          continue;

        vec_t pts(x_end - x_start);
        for (int64_t i = 0; i < x_end - x_start; i++) {
          pts[i].resize(x.size(1));
          for (int64_t j = 0; j < x.size(1); j++) {
            pts[i][j] = x_data[(i + x_start) * x.size(1) + j];
          }
        }

        typedef KDTreeVectorOfVectorsAdaptor<vec_t, scalar_t> my_kd_tree_t;

        my_kd_tree_t mat_index(x.size(1), pts, 10);
        mat_index.index->buildIndex();

        for (int64_t i = y_start; i < y_end; i++) {
          std::vector<std::pair<size_t, scalar_t>> ret_matches;
          size_t num_matches = mat_index.index->radiusSearch(
              y_data + i * y.size(1), r * r, ret_matches, params);

          for (size_t j = 0;
               j < std::min(num_matches, (size_t)max_num_neighbors); j++) {
            out_vec.push_back(x_start + ret_matches[j].first);
            out_vec.push_back(i);
          }
        }
      }
    }
  });

  const int64_t size = out_vec.size() / 2;
  auto out = torch::from_blob(out_vec.data(), {size, 2},
                              x.options().dtype(torch::kLong));
  return out.t().index_select(0, torch::tensor({1, 0}));
}
