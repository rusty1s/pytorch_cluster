#include <torch/extension.h>

at::Tensor neighbor_sampler(at::Tensor start, at::Tensor cumdeg, size_t size,
                            float factor) {

  auto start_ptr = start.data<int64_t>();
  auto cumdeg_ptr = cumdeg.data<int64_t>();

  std::vector<int64_t> e_ids;
  for (ptrdiff_t i = 0; i < start.size(0); i++) {
    int64_t low = cumdeg_ptr[start_ptr[i]];
    int64_t high = cumdeg_ptr[start_ptr[i] + 1];
    size_t num_neighbors = high - low;

    size_t size_i = size_t(ceil(factor * float(num_neighbors)));
    size_i = (size_i < size) ? size_i : size;

    // If the number of neighbors is approximately equal to the number of
    // neighbors which are requested, we use `randperm` to sample without
    // replacement, otherwise we sample random numbers into a set as long as
    // necessary.
    std::unordered_set<int64_t> set;
    if (size_i < 0.7 * float(num_neighbors)) {
      while (set.size() < size_i) {
        int64_t z = rand() % num_neighbors;
        set.insert(z + low);
      }
      std::vector<int64_t> v(set.begin(), set.end());
      e_ids.insert(e_ids.end(), v.begin(), v.end());
    } else {
      auto sample = at::randperm(num_neighbors, start.options());
      auto sample_ptr = sample.data<int64_t>();
      for (size_t j = 0; j < size_i; j++) {
        e_ids.push_back(sample_ptr[j] + low);
      }
    }
  }

  int64_t len = e_ids.size();
  auto e_id = torch::from_blob(e_ids.data(), {len}, start.options()).clone();
  return e_id;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("neighbor_sampler", &neighbor_sampler, "Neighbor Sampler (CPU)");
}
