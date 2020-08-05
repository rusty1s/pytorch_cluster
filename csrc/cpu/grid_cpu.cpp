#include "grid_cpu.h"

#include "utils.h"

torch::Tensor grid_cpu(torch::Tensor pos, torch::Tensor size,
                       torch::optional<torch::Tensor> optional_start,
                       torch::optional<torch::Tensor> optional_end) {

  CHECK_CPU(pos);
  CHECK_CPU(size);

  if (optional_start.has_value())
    CHECK_CPU(optional_start.value());
  if (optional_start.has_value())
    CHECK_CPU(optional_start.value());

  pos = pos.view({pos.size(0), -1});
  CHECK_INPUT(size.numel() == pos.size(1));

  if (!optional_start.has_value())
    optional_start = std::get<0>(pos.min(0));
  else
    CHECK_INPUT(optional_start.value().numel() == pos.size(1));

  if (!optional_end.has_value())
    optional_end = std::get<0>(pos.max(0));
  else
    CHECK_INPUT(optional_start.value().numel() == pos.size(1));

  auto start = optional_start.value();
  auto end = optional_end.value();

  pos = pos - start.unsqueeze(0);

  auto num_voxels = (end - start).true_divide(size).toType(torch::kLong) + 1;
  num_voxels = num_voxels.cumprod(0);
  num_voxels =
      torch::cat({torch::ones(1, num_voxels.options()), num_voxels}, 0);
  num_voxels = num_voxels.narrow(0, 0, size.size(0));

  auto out = pos.true_divide(size.view({1, -1})).toType(torch::kLong);
  out *= num_voxels.view({1, -1});
  out = out.sum(1);

  return out;
}
