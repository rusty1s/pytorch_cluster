#pragma once

#include <ATen/ATen.h>

std::tuple<at::Tensor, at::Tensor> remove_self_loops(at::Tensor row,
                                                     at::Tensor col) {
  auto mask = row != col;
  return std::make_tuple(row.masked_select(mask), col.masked_select(mask));
}

std::tuple<at::Tensor, at::Tensor, at::Tensor>
remove_self_loops(at::Tensor row, at::Tensor col, at::Tensor weight) {
  auto mask = row != col;
  return std::make_tuple(row.masked_select(mask), col.masked_select(mask),
                         weight.masked_select(mask));
}

std::tuple<at::Tensor, at::Tensor> rand(at::Tensor row, at::Tensor col) {
  auto perm = at::empty(row.size(0), row.options());
  at::randperm_out(perm, row.size(0));
  return std::make_tuple(row.index_select(0, perm), col.index_select(0, perm));
}

std::tuple<at::Tensor, at::Tensor> sort_by_row(at::Tensor row, at::Tensor col) {
  at::Tensor perm;
  std::tie(row, perm) = row.sort();
  return std::make_tuple(row, col.index_select(0, perm));
}

std::tuple<at::Tensor, at::Tensor, at::Tensor>
sort_by_row(at::Tensor row, at::Tensor col, at::Tensor weight) {
  at::Tensor perm;
  std::tie(row, perm) = row.sort();
  return std::make_tuple(row, col.index_select(0, perm),
                         weight.index_select(0, perm));
}

at::Tensor degree(at::Tensor row, int64_t num_nodes) {
  auto zero = at::zeros(num_nodes, row.options());
  auto one = at::ones(row.size(0), row.options());
  return zero.scatter_add_(0, row, one);
}

std::tuple<at::Tensor, at::Tensor> to_csr(at::Tensor row, at::Tensor col,
                                          int64_t num_nodes) {
  std::tie(row, col) = sort_by_row(row, col);
  row = degree(row, num_nodes).cumsum(0);
  row = at::cat({at::zeros(1, row.options()), row}, 0); // Prepend zero.
  return std::make_tuple(row, col);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor>
to_csr(at::Tensor row, at::Tensor col, at::Tensor weight, int64_t num_nodes) {
  std::tie(row, col, weight) = sort_by_row(row, col, weight);
  row = degree(row, num_nodes).cumsum(0);
  row = at::cat({at::zeros(1, row.options()), row}, 0); // Prepend zero.
  return std::make_tuple(row, col, weight);
}
