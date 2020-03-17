#include "sampler_cpu.h"

#include "utils.h"

torch::Tensor neighbor_sampler_cpu(torch::Tensor start, torch::Tensor rowptr,
                                   int64_t count, double factor) {

  auto start_data = start.data_ptr<int64_t>();
  auto rowptr_data = rowptr.data_ptr<int64_t>();

  std::vector<int64_t> e_ids;
  for (auto i = 0; i < start.size(0); i++) {
    auto row_start = rowptr_data[start_data[i]];
    auto row_end = rowptr_data[start_data[i] + 1];
    auto num_neighbors = row_end - row_start;

    int64_t size = count;
    if (count < 1)
      size = int64_t(ceil(factor * float(num_neighbors)));
    if (size > num_neighbors)
      size = num_neighbors;

    // If the number of neighbors is approximately equal to the number of
    // neighbors which are requested, we use `randperm` to sample without
    // replacement, otherwise we sample random numbers into a set as long
    // as necessary.
    std::unordered_set<int64_t> set;
    if (size < 0.7 * float(num_neighbors)) {
      while (int64_t(set.size()) < size) {
        int64_t sample = rand() % num_neighbors;
        set.insert(sample + row_start);
      }
      std::vector<int64_t> v(set.begin(), set.end());
      e_ids.insert(e_ids.end(), v.begin(), v.end());
    } else {
      auto sample = torch::randperm(num_neighbors, start.options());
      auto sample_data = sample.data_ptr<int64_t>();
      for (auto j = 0; j < size; j++) {
        e_ids.push_back(sample_data[j] + row_start);
      }
    }
  }

  int64_t length = e_ids.size();
  return torch::from_blob(e_ids.data(), {length}, start.options()).clone();
}
