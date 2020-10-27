#include "rw_cpu.h"

#include <ATen/ParallelOpenMP.h>

#include "utils.h"

torch::Tensor random_walk_cpu(torch::Tensor rowptr, torch::Tensor col,
                              torch::Tensor start, int64_t walk_length,
                              double p, double q) {
  CHECK_CPU(rowptr);
  CHECK_CPU(col);
  CHECK_CPU(start);

  CHECK_INPUT(rowptr.dim() == 1);
  CHECK_INPUT(col.dim() == 1);
  CHECK_INPUT(start.dim() == 1);

  auto rand = torch::rand({start.size(0), walk_length},
                          start.options().dtype(torch::kFloat));

  auto out = torch::empty({start.size(0), walk_length + 1}, start.options());

  auto rowptr_data = rowptr.data_ptr<int64_t>();
  auto col_data = col.data_ptr<int64_t>();
  auto start_data = start.data_ptr<int64_t>();
  auto rand_data = rand.data_ptr<float>();
  auto out_data = out.data_ptr<int64_t>();

  int64_t grain_size = at::internal::GRAIN_SIZE / walk_length;
  at::parallel_for(0, start.numel(), grain_size, [&](int64_t b, int64_t e) {
    for (auto n = b; n < e; n++) {
      auto cur = start_data[n];
      auto offset = n * (walk_length + 1);
      out_data[offset] = cur;

      int64_t row_start, row_end, rnd;
      for (auto l = 1; l <= walk_length; l++) {
        row_start = rowptr_data[cur], row_end = rowptr_data[cur + 1];
        if (row_end - row_start == 0) {
          cur = n;
        } else {
          rnd = int64_t(rand_data[n * walk_length + (l - 1)] *
                        (row_end - row_start));
          cur = col_data[row_start + rnd];
        }
        out_data[offset + l] = cur;
      }
    }
  });

  return out;
}
