#include "knn_cpu.h"

#include "utils.h"
#include "utils/neighbors.cpp"

torch::Tensor knn_cpu(torch::Tensor x, torch::Tensor y,
                      torch::optional<torch::Tensor> ptr_x,
                      torch::optional<torch::Tensor> ptr_y, int64_t k,
                      int64_t num_workers) {

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

  std::vector<size_t> *out_vec = new std::vector<size_t>();

  AT_DISPATCH_ALL_TYPES(x.scalar_type(), "radius_cpu", [&] {
    auto x_data = x.data_ptr<scalar_t>();
    auto y_data = y.data_ptr<scalar_t>();
    auto x_vec = std::vector<scalar_t>(x_data, x_data + x.numel());
    auto y_vec = std::vector<scalar_t>(y_data, y_data + y.numel());

    if (!ptr_x.has_value()) {
      nanoflann_neighbors<scalar_t>(y_vec, x_vec, out_vec, 0, x.size(-1), 0,
                                    num_workers, k, 0);
    } else {
      auto sx = (ptr_x.value().narrow(0, 1, ptr_x.value().numel() - 1) -
                 ptr_x.value().narrow(0, 0, ptr_x.value().numel() - 1));
      auto sy = (ptr_y.value().narrow(0, 1, ptr_y.value().numel() - 1) -
                 ptr_y.value().narrow(0, 0, ptr_y.value().numel() - 1));
      auto sx_data = sx.data_ptr<int64_t>();
      auto sy_data = sy.data_ptr<int64_t>();
      auto sx_vec = std::vector<long>(sx_data, sx_data + sx.numel());
      auto sy_vec = std::vector<long>(sy_data, sy_data + sy.numel());
      batch_nanoflann_neighbors<scalar_t>(y_vec, x_vec, sy_vec, sx_vec, out_vec,
                                          k, x.size(-1), 0, k, 0);
    }
  });

  const int64_t size = out_vec->size() / 2;
  auto out = torch::from_blob(out_vec->data(), {size, 2},
                              x.options().dtype(torch::kLong));
  return out.t().index_select(0, torch::tensor({1, 0}));
}
