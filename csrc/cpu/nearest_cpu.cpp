#include <algorithm>

#include "nearest_cpu.h"

#include <ATen/Parallel.h>

#include "utils.h"

torch::Tensor dists(const torch::Tensor &x, const torch::Tensor &y) {
  if (x.dtype() != c10::ScalarType::Half && y.dtype() != c10::ScalarType::Half)
    return torch::cdist(x, y);

  // Get the sizes of x and y.
  int64_t n = x.size(0);
  int64_t m = y.size(0);

  // Initialize the distances.
  torch::Tensor distances = torch::zeros({n, m});

  // Calculate the distances.
  const int64_t grain_size = 1;
  at::parallel_for(0, n, grain_size, [&](int64_t begin, int64_t end) {
    const auto calcDistances = [&](torch::Tensor &out, int64_t offset = 0) {
      for (int idx = begin; idx<end; idx++) {
        const auto sqr_dist = torch::pow(x[idx] - y, 2).sum(1);
        out.index_put_({idx-offset}, sqr_dist);
      }
    };
    const auto size = end - begin;
    if (size > 1) {
      torch::Tensor distances_chunk = torch::zeros({size, m});
      calcDistances(distances_chunk, begin);
      distances.slice(0, begin, end) = distances_chunk;
    } else {
      calcDistances(distances);
    }
  });

  // Return the distances.
  return distances;
}

torch::Tensor nearest_cpu(torch::Tensor x, torch::Tensor y,
  torch::Tensor batch_x, torch::Tensor batch_y) {
  CHECK_CPU(x);
  CHECK_CPU(y);
  CHECK_CPU(batch_x);
  CHECK_CPU(batch_y);

  batch_x = batch_x.contiguous();
  batch_y = batch_y.contiguous();

  if (batch_x.size(0) && batch_y.size(0)) {
    const auto unique_batch_x = std::get<0>(at::unique_consecutive(batch_x));
    const auto unique_batch_y = std::get<0>(at::unique_consecutive(batch_y));
    if (!torch::equal(unique_batch_x, unique_batch_y))
      throw std::invalid_argument(
                    "Some batch indices occur in 'batch_x' "
                    "that do not occur in 'batch_y'");

    if( (x.dim() != 2 || batch_x.dim() != 1) ||
        (y.dim() != 2 || batch_y.dim() != 1) ||
        x.size(0) != batch_x.size(0) ||
        y.size(0) != batch_y.size(0) )
      throw std::invalid_argument("");

    const auto min_xy = at::minimum(x.min(), y.min());
    x = at::sub(x, min_xy);
    y = at::sub(y, min_xy);

    const auto max_xy = at::maximum(x.max(), y.max());
    x = at::div(x, max_xy);
    y = at::div(y, max_xy);

    const double D = x.size(x.dim()-1);
    const auto batch_x_view = batch_x.view({-1, 1}).to(x.dtype());
    const auto batch_x_rescaled = x.mul(D);
    x = at::cat({x, batch_x_rescaled}, x.dim()-1);
    const auto batch_y_view = batch_y.view({-1, 1}).to(y.dtype());
    const auto batch_y_rescaled = y.mul(D);
    y = at::cat({y, batch_y_rescaled}, y.dim()-1);
  }

  const auto distances = dists(x, y);
  return at::argmin(distances, 1);
}
