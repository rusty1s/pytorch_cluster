#include <torch/extension.h>

at::Tensor grid(at::Tensor pos, at::Tensor size, at::Tensor start,
                at::Tensor end) {
  pos = pos - start.view({1, -1});

  auto num_voxels = ((end - start) / size).toType(at::kLong) + 1;
  num_voxels = num_voxels.cumprod(0);

  num_voxels = at::cat({at::ones(1, num_voxels.options()), num_voxels}, 0);
  auto index = at::empty(size.size(0), num_voxels.options());
  at::arange_out(index, size.size(0));
  num_voxels = num_voxels.index_select(0, index);

  auto cluster = (pos / size.view({1, -1})).toType(at::kLong);
  cluster *= num_voxels.view({1, -1});
  cluster = cluster.sum(1);

  return cluster;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("grid", &grid, "Grid (CPU)"); }
