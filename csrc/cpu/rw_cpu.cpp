#include "rw_cpu.h"

#include <ATen/Parallel.h>

#include "utils.h"

void uniform_sampling(const int64_t *rowptr, const int64_t *col,
                      const int64_t *start, int64_t *n_out, int64_t *e_out,
                      const int64_t numel, const int64_t walk_length) {

  auto rand = torch::rand({numel, walk_length});
  auto rand_data = rand.data_ptr<float>();

  int64_t grain_size = at::internal::GRAIN_SIZE / walk_length;
  at::parallel_for(0, numel, grain_size, [&](int64_t begin, int64_t end) {
    for (auto n = begin; n < end; n++) {
      int64_t n_cur = start[n], e_cur, row_start, row_end, idx;

      n_out[n * (walk_length + 1)] = n_cur;

      for (auto l = 0; l < walk_length; l++) {
        row_start = rowptr[n_cur], row_end = rowptr[n_cur + 1];
        if (row_end - row_start == 0) {
          e_cur = -1;
        } else {
          idx = int64_t(rand_data[n * walk_length + l] * (row_end - row_start));
          e_cur = row_start + idx;
          n_cur = col[e_cur];
        }
        n_out[n * (walk_length + 1) + (l + 1)] = n_cur;
        e_out[n * walk_length + l] = e_cur;
      }
    }
  });
}

bool inline is_neighbor(const int64_t *rowptr, const int64_t *col, int64_t v,
                        int64_t w) {
  int64_t row_start = rowptr[v], row_end = rowptr[v + 1];
  for (auto i = row_start; i < row_end; i++) {
    if (col[i] == w)
      return true;
  }
  return false;
}

// See: https://louisabraham.github.io/articles/node2vec-sampling.html
void rejection_sampling(const int64_t *rowptr, const int64_t *col,
                        int64_t *start, int64_t *n_out, int64_t *e_out,
                        const int64_t numel, const int64_t walk_length,
                        const double p, const double q) {

  double max_prob = fmax(fmax(1. / p, 1.), 1. / q);
  double prob_0 = 1. / p / max_prob;
  double prob_1 = 1. / max_prob;
  double prob_2 = 1. / q / max_prob;

  int64_t grain_size = at::internal::GRAIN_SIZE / walk_length;
  at::parallel_for(0, numel, grain_size, [&](int64_t begin, int64_t end) {
    for (auto n = begin; n < end; n++) {
      int64_t t = start[n], v, x, e_cur, row_start, row_end;

      n_out[n * (walk_length + 1)] = t;

      row_start = rowptr[t], row_end = rowptr[t + 1];
      if (row_end - row_start == 0) {
        e_cur = -1;
        v = t;
      } else {
        e_cur = row_start + (rand() % (row_end - row_start));
        v = col[e_cur];
      }
      n_out[n * (walk_length + 1) + 1] = v;
      e_out[n * walk_length] = e_cur;

      for (auto l = 1; l < walk_length; l++) {
        row_start = rowptr[v], row_end = rowptr[v + 1];

        if (row_end - row_start == 0) {
          e_cur = -1;
          x = v;
        } else if (row_end - row_start == 1) {
          e_cur = row_start;
          x = col[e_cur];
        } else {
          while (true) {
            e_cur = row_start + (rand() % (row_end - row_start));
            x = col[e_cur];

            auto r = ((double)rand() / (RAND_MAX)); // [0, 1)

            if (x == t && r < prob_0)
              break;
            else if (is_neighbor(rowptr, col, x, t) && r < prob_1)
              break;
            else if (r < prob_2)
              break;
          }
        }

        n_out[n * (walk_length + 1) + (l + 1)] = x;
        e_out[n * walk_length + l] = e_cur;
        t = v;
        v = x;
      }
    }
  });
}

std::tuple<torch::Tensor, torch::Tensor>
random_walk_cpu(torch::Tensor rowptr, torch::Tensor col, torch::Tensor start,
                int64_t walk_length, double p, double q) {
  CHECK_CPU(rowptr);
  CHECK_CPU(col);
  CHECK_CPU(start);

  CHECK_INPUT(rowptr.dim() == 1);
  CHECK_INPUT(col.dim() == 1);
  CHECK_INPUT(start.dim() == 1);

  auto n_out = torch::empty({start.size(0), walk_length + 1}, start.options());
  auto e_out = torch::empty({start.size(0), walk_length}, start.options());

  auto rowptr_data = rowptr.data_ptr<int64_t>();
  auto col_data = col.data_ptr<int64_t>();
  auto start_data = start.data_ptr<int64_t>();
  auto n_out_data = n_out.data_ptr<int64_t>();
  auto e_out_data = e_out.data_ptr<int64_t>();

  if (p == 1. && q == 1.) {
    uniform_sampling(rowptr_data, col_data, start_data, n_out_data, e_out_data,
                     start.numel(), walk_length);
  } else {
    rejection_sampling(rowptr_data, col_data, start_data, n_out_data,
                       e_out_data, start.numel(), walk_length, p, q);
  }

  return std::make_tuple(n_out, e_out);
}


void compute_cdf(const int64_t *rowptr, const float_t *edge_weight,
		 float_t *edge_weight_cdf, int64_t numel) {
  /* Convert edge weights to CDF as given in [1]

  [1] https://github.com/louisabraham/fastnode2vec/blob/master/fastnode2vec/graph.py#L148
  */
  at::parallel_for(0, numel - 1, at::internal::GRAIN_SIZE, [&](int64_t begin, int64_t end) {
    for(int64_t i = begin; i < end; i++) {
      int64_t row_start = rowptr[i], row_end = rowptr[i + 1];
      float_t acc = 0.0;

      for(int64_t j = row_start; j < row_end; j++) {
        acc += edge_weight[j];
        edge_weight_cdf[j] = acc;
      }
    }
  });
}


int64_t get_offset(const float_t *edge_weight, int64_t start, int64_t end) {
  /*
  The implementation given in [1] utilizes the `searchsorted` function in Numpy.
  It is also available in PyTorch and its C++ API (via `at::searchsorted()`).
  However, the implementation is adopted to the general case where the searched
  values can be a multidimensional tensor. In our case, we have a 1D tensor of
  edge weights (in form of a Cumulative Distribution Function) and a single
  value, whose position we want to compute. To eliminate the overhead introduced
  in the PyTorch implementation, one can examine the source code of
  `searchsorted` [2] and find that for our case the whole function call can be
  reduced to calling the `cus_lower_bound()` function. Unfortunately, we cannot
  access it directly (the namespace is not exposed to the public API), but the
  implementation is just a simple binary search. The code was copied here and
  reduced to the bare minimum.

  [1] https://github.com/louisabraham/fastnode2vec/blob/master/fastnode2vec/graph.py#L69
  [2] https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/Bucketization.cpp
  */
  float_t value = ((float_t)rand() / RAND_MAX); // [0, 1)
  int64_t original_start = start;

  while (start < end) {
    const int64_t mid = start + ((end - start) >> 1);
    const float_t mid_val = edge_weight[mid];
    if (!(mid_val >= value)) {
      start = mid + 1;
    }
    else {
      end = mid;
    }
  }

  return start - original_start;
}

// See: https://louisabraham.github.io/articles/node2vec-sampling.html
// See also: https://github.com/louisabraham/fastnode2vec/blob/master/fastnode2vec/graph.py#L69
void rejection_sampling_weighted(const int64_t *rowptr, const int64_t *col,
                                 const float_t *edge_weight_cdf, int64_t *start,
                                 int64_t *n_out, int64_t *e_out,
                                 const int64_t numel, const int64_t walk_length,
                                 const double p, const double q) {

  double max_prob = fmax(fmax(1. / p, 1.), 1. / q);
  double prob_0 = 1. / p / max_prob;
  double prob_1 = 1. / max_prob;
  double prob_2 = 1. / q / max_prob;

  int64_t grain_size = at::internal::GRAIN_SIZE / walk_length;
  at::parallel_for(0, numel, grain_size, [&](int64_t begin, int64_t end) {
    for (auto n = begin; n < end; n++) {
      int64_t t = start[n], v, x, e_cur, row_start, row_end;

      n_out[n * (walk_length + 1)] = t;

      row_start = rowptr[t], row_end = rowptr[t + 1];

      if (row_end - row_start == 0) {
        e_cur = -1;
        v = t;
      } else {
        e_cur = row_start + get_offset(edge_weight_cdf, row_start, row_end);
        v = col[e_cur];
      }
      n_out[n * (walk_length + 1) + 1] = v;
      e_out[n * walk_length] = e_cur;

      for (auto l = 1; l < walk_length; l++) {
        row_start = rowptr[v], row_end = rowptr[v + 1];

        if (row_end - row_start == 0) {
          e_cur = -1;
          x = v;
        } else if (row_end - row_start == 1) {
          e_cur = row_start;
          x = col[e_cur];
        } else {
          if (p == 1. && q == 1.) {
            e_cur = row_start + get_offset(edge_weight_cdf, row_start, row_end);
            x = col[e_cur];
          }
          else {
            while (true) {
              e_cur = row_start + get_offset(edge_weight_cdf, row_start, row_end);
              x = col[e_cur];

              auto r = ((double)rand() / (RAND_MAX)); // [0, 1)

              if (x == t && r < prob_0)
                break;
              else if (is_neighbor(rowptr, col, x, t) && r < prob_1)
                break;
              else if (r < prob_2)
                break;
            }
          }
        }

        n_out[n * (walk_length + 1) + (l + 1)] = x;
        e_out[n * walk_length + l] = e_cur;
        t = v;
        v = x;
      }
    }
  });
}


std::tuple<torch::Tensor, torch::Tensor>
random_walk_weighted_cpu(torch::Tensor rowptr, torch::Tensor col,
                         torch::Tensor edge_weight, torch::Tensor start,
                         int64_t walk_length, double p, double q) {
  CHECK_CPU(rowptr);
  CHECK_CPU(col);
  CHECK_CPU(edge_weight);
  CHECK_CPU(start);

  CHECK_INPUT(rowptr.dim() == 1);
  CHECK_INPUT(col.dim() == 1);
  CHECK_INPUT(edge_weight.dim() == 1);
  CHECK_INPUT(start.dim() == 1);

  auto n_out = torch::empty({start.size(0), walk_length + 1}, start.options());
  auto e_out = torch::empty({start.size(0), walk_length}, start.options());

  auto rowptr_data = rowptr.data_ptr<int64_t>();
  auto col_data = col.data_ptr<int64_t>();
  auto edge_weight_data = edge_weight.data_ptr<float_t>();
  auto start_data = start.data_ptr<int64_t>();
  auto n_out_data = n_out.data_ptr<int64_t>();
  auto e_out_data = e_out.data_ptr<int64_t>();

  auto edge_weight_cdf = torch::empty({edge_weight.size(0)}, edge_weight.options());
  auto edge_weight_cdf_data = edge_weight_cdf.data_ptr<float_t>();

  compute_cdf(rowptr_data, edge_weight_data, edge_weight_cdf_data, rowptr.numel());

  rejection_sampling_weighted(rowptr_data, col_data, edge_weight_cdf_data,
                              start_data, n_out_data, e_out_data, start.numel(),
                              walk_length, p, q);

  return std::make_tuple(n_out, e_out);
}
