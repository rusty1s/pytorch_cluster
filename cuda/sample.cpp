#include <torch/torch.h>
#define IS_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be CUDA tensor");
#define IS_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " is not contiguous");
#define CHECK_INPUT(x) IS_CUDA(x) IS_CONTIGUOUS(x)

std::vector<at::Tensor> query_radius_cuda(
    int batch_size,
    at::Tensor batch_slices,
    at::Tensor query_batch_slices,
    at::Tensor pos,
    at::Tensor query_pos,
    double radius,
    int max_num_neighbors,
    bool include_self);



std::vector<at::Tensor> query_radius(
    int batch_size,
    at::Tensor batch_slices,
    at::Tensor query_batch_slices,
    at::Tensor pos,
    at::Tensor query_pos,
    double radius,
    int max_num_neighbors,
    bool include_self) {
  CHECK_INPUT(batch_slices);
  CHECK_INPUT(query_batch_slices);
  CHECK_INPUT(pos);
  CHECK_INPUT(query_pos);

  return query_radius_cuda(batch_size, batch_slices, query_batch_slices, pos, query_pos, radius, max_num_neighbors, include_self);
}

std::vector<at::Tensor> query_knn_cuda(
    int batch_size,
    at::Tensor batch_slices,
    at::Tensor query_batch_slices,
    at::Tensor pos,
    at::Tensor query_pos,
    int num_neighbors,
    bool include_self);

std::vector<at::Tensor> query_knn(
    int batch_size,
    at::Tensor batch_slices,
    at::Tensor query_batch_slices,
    at::Tensor pos,
    at::Tensor query_pos,
    int num_neighbors,
    bool include_self) {
  CHECK_INPUT(batch_slices);
  CHECK_INPUT(query_batch_slices);
  CHECK_INPUT(pos);
  CHECK_INPUT(query_pos);

  return query_knn_cuda(batch_size, batch_slices, query_batch_slices, pos, query_pos, num_neighbors, include_self);
}

at::Tensor farthest_point_sampling_cuda(
    int batch_size,
    at::Tensor batch_slices,
    at::Tensor pos,
    int num_sample,
    at::Tensor start_points);

at::Tensor farthest_point_sampling(
    int batch_size,
    at::Tensor batch_slices,
    at::Tensor pos,
    int num_sample,
    at::Tensor start_points) {
  CHECK_INPUT(batch_slices);
  CHECK_INPUT(pos);
  CHECK_INPUT(start_points);

  return farthest_point_sampling_cuda(batch_size, batch_slices, pos, num_sample, start_points);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("query_radius", &query_radius, "Query Radius (CUDA)");
  m.def("query_knn", &query_knn, "Query K-Nearest neighbor (CUDA)");
  m.def("farthest_point_sampling", &farthest_point_sampling, "Farthest Point Sampling (CUDA)");
}




