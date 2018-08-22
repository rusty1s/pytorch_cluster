#pragma once

#include <torch/torch.h>

std::tuple<at::Tensor, at::Tensor> remove_self_loops(at::Tensor row,
                                                     at::Tensor col) {
  auto mask = row != col;
  return make_tuple(row.masked_select(mask), col.masked_select(mask));
}

std::tuple<at::Tensor, at::Tensor, at::Tensor>
remove_self_loops(at::Tensor row, at::Tensor col, at::Tensor weight) {
  auto mask = row != col;
  return make_tuple(row.masked_select(mask), col.masked_select(mask),
                    weight.masked_select(mask));
}

at::Tensor randperm(int64_t n) {
  auto out = at::empty(n, torch::CPU(at::kLong));
  at::randperm_out(out, n);
  return out;
}

std::tuple<at::Tensor, at::Tensor> rand(at::Tensor row, at::Tensor col) {
  auto perm = randperm(row.size(0));
  return make_tuple(row.index_select(perm), col.index_select(perm));
}

std::tuple<at::Tensor, at::Tensor> sort_by_row(at::Tensor row, at::Tensor col) {
  Tensor perm;
  tie(row, perm) = row.sort();
  col = col.index_select(0, perm);
  return stack({row, col}, 0);
}

inline Tensor degree(Tensor row, int64_t num_nodes) {
  auto zero = zeros(num_nodes, row.type());
  auto one = ones(row.size(0), row.type());
  return zero.scatter_add_(0, row, one);
}

inline tuple<Tensor, Tensor> to_csr(Tensor index, int64_t num_nodes) {
  index = sort_by_row(index);
  auto row = degree(index[0], num_nodes).cumsum(0);
  row = cat({zeros(1, row.type()), row}, 0); // Prepend zero.
  return make_tuple(row, index[1]);
}

// std::tie(row, col) = randperm(row, col);
// std::tie(row, col) = to_csr(row, col);
