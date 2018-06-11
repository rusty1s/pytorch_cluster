#include <ATen/ATen.h>

#include "color.cuh"
#include "common.cuh"

at::Tensor graclus(at::Tensor row, at::Tensor col, int num_nodes) {
  // Remove self-loops.
  auto mask = row != col;
  row = row.masked_select(mask);
  col.masked_select(mask);

  // Sort by row index.
  at::Tensor perm;
  std::tie(row, perm) = row.sort();
  col = col.index_select(0, perm);

  // Generate helper vectors.
  auto cluster = at::full(row.type(), {num_nodes}, -1);
  auto prop = at::full(row.type(), {num_nodes}, -1);
  auto deg = degree(row, num_nodes);
  auto cum_deg = deg.cumsum(0);

  color(cluster);

  /* while (!color(cluster)) { */
  /*   propose(cluster, prop, row, col, weight, deg, cum_deg); */
  /*   response(cluster, prop, row, col, weight, deg, cum_deg); */
  /* } */

  return cluster;
}

at::Tensor weighted_graclus(at::Tensor row, at::Tensor col, at::Tensor weight,
                            int num_nodes) {
  // Remove self-loops.
  auto mask = row != col;
  row = row.masked_select(mask);
  col = col.masked_select(mask);
  weight = weight.masked_select(mask);

  // Sort by row index.
  at::Tensor perm;
  std::tie(row, perm) = row.sort();
  col = col.index_select(0, perm);
  weight = weight.index_select(0, perm);

  // Generate helper vectors.
  auto cluster = at::full(row.type(), {num_nodes}, -1);
  auto prop = at::full(row.type(), {num_nodes}, -1);
  auto deg = degree(row, num_nodes);
  auto cum_deg = deg.cumsum(0);

  color(cluster);

  /* while (!color(cluster)) { */
  /*   weighted_propose(cluster, prop, row, col, weight, deg, cum_deg); */
  /*   weighted_response(cluster, prop, row, col, weight, deg, cum_deg); */
  /* } */

  return cluster;
}
