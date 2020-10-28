#include "rw_cpu.h"

#include <ATen/Parallel.h>

#include "utils.h"

std::tuple<torch::Tensor, torch::Tensor>
random_walk_cpu(torch::Tensor rowptr, torch::Tensor col, torch::Tensor start,
                int64_t walk_length, double p, double q) {
  CHECK_CPU(rowptr);
  CHECK_CPU(col);
  CHECK_CPU(start);

  CHECK_INPUT(rowptr.dim() == 1);
  CHECK_INPUT(col.dim() == 1);
  CHECK_INPUT(start.dim() == 1);

  auto rand = torch::rand({start.size(0), walk_length},
                          start.options().dtype(torch::kFloat));

  auto n_out = torch::empty({start.size(0), walk_length + 1}, start.options());
  auto e_out = torch::empty({start.size(0), walk_length}, start.options());

  auto rowptr_data = rowptr.data_ptr<int64_t>();
  auto col_data = col.data_ptr<int64_t>();
  auto start_data = start.data_ptr<int64_t>();
  auto rand_data = rand.data_ptr<float>();
  auto n_out_data = n_out.data_ptr<int64_t>();
  auto e_out_data = e_out.data_ptr<int64_t>();

  int64_t grain_size = at::internal::GRAIN_SIZE / walk_length;
  at::parallel_for(0, start.numel(), grain_size, [&](int64_t b, int64_t e) {
    for (auto n = b; n < e; n++) {
      int64_t n_cur = start_data[n], e_cur, row_start, row_end, rnd;

      n_out_data[n * (walk_length + 1)] = n_cur;

      for (auto l = 0; l < walk_length; l++) {
        row_start = rowptr_data[n_cur], row_end = rowptr_data[n_cur + 1];
        if (row_end - row_start == 0) {
          e_cur = -1;
        } else {
          rnd = int64_t(rand_data[n * walk_length + l] * (row_end - row_start));
          e_cur = row_start + rnd;
          n_cur = col_data[e_cur];
        }
        n_out_data[n * (walk_length + 1) + (l + 1)] = n_cur;
        e_out_data[n * walk_length + l] = e_cur;
      }
    }
  });

  return std::make_tuple(n_out, e_out);
}
