#include <torch/torch.h>


inline std::tuple<at::Tensor, at::Tensor> randperm(at::Tensor row, at::Tensor col) {
  /* at::Tensor perm; */
  /* std::tie(row, perm) = row.sort(); */
  /* col = col.index_select(0, perm); */

  /* TODO: randperm */
  /* TODO: randperm_sort_row */
  return { row, col };
}


inline at::Tensor degree(at::Tensor index, int64_t num_nodes) {
  auto zero = at::zeros(torch::CPU(at::kLong), { num_nodes });
  return zero.scatter_add_(0, index, at::ones_like(index));
}


at::Tensor graclus(at::Tensor row, at::Tensor col, int64_t num_nodes) {
  std::tie(row, col) = randperm(row, col);
  auto deg = degree(row, num_nodes);
  auto cluster = at::empty(torch::CPU(at::kLong), { num_nodes }).fill_(-1);

  auto *row_data = row.data<int64_t>();
  auto *col_data = col.data<int64_t>();
  auto *deg_data = deg.data<int64_t>();
  auto *cluster_data = cluster.data<int64_t>();

  int64_t n_idx = 0, e_idx = 0, d_idx, r, c;
  while (e_idx < row.size(0)) {
    r = row_data[e_idx];
    if (cluster_data[r] < 0) {
      cluster_data[r] = r;
      for (d_idx = 0; d_idx < deg_data[r]; d_idx++) {
        c = col_data[e_idx + d_idx];
        if (cluster_data[c] < 0) {
          cluster_data[r] = std::min(r, c);
          cluster_data[c] = std::min(r, c);
          break;
        }
      }
    }
    e_idx += deg_data[n_idx];
    n_idx++;
  }

  return cluster;
}


at::Tensor grid(at::Tensor pos, at::Tensor size, at::Tensor start, at::Tensor end) {
  size = size.toType(pos.type());
  start = start.toType(pos.type());
  end = end.toType(pos.type());

  pos = pos - start.view({ 1, -1 });
  auto num_voxels = ((end - start) / size).toType(at::kLong);
  num_voxels = (num_voxels + 1).cumsum(0);
  num_voxels -= num_voxels.data<int64_t>()[0];
  num_voxels.data<int64_t>()[0] = 1;

  auto cluster = pos / size.view({ 1, -1 });
  cluster = cluster.toType(at::kLong);
  cluster *= num_voxels.view({ 1, -1 });
  cluster = cluster.sum(1);

  return cluster;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("graclus", &graclus, "Graclus (CPU)");
  m.def("grid", &grid, "Grid (CPU)");
}
