#include <ATen/ATen.h>

#include "atomics.cuh"
#include "utils.cuh"

#define THREADS 1024

template <typename scalar_t>
__global__ void
fps_kernel(scalar_t *__restrict__ x, int64_t *__restrict__ cum_deg,
           int64_t *__restrict__ cum_k, int64_t *__restrict__ start,
           scalar_t *__restrict__ dist, scalar_t *__restrict__ tmp_dist,
           int64_t *__restrict__ out, size_t dim) {

  const size_t batch_idx = blockIdx.x;
  const size_t idx = threadIdx.x;
  const size_t stride = blockDim.x; // == THREADS

  const size_t start_idx = cum_deg[batch_idx];
  const size_t end_idx = cum_deg[batch_idx + 1];

  int64_t old = start_idx + start[batch_idx];

  if (idx == 0) {
    out[cum_k[batch_idx]] = old;
  }

  for (ptrdiff_t m = cum_k[batch_idx] + 1; m < cum_k[batch_idx + 1]; m++) {

    for (ptrdiff_t n = start_idx + idx; n < end_idx; n += stride) {
      tmp_dist[n] = 0;
    }

    __syncthreads();
    for (ptrdiff_t i = start_idx * dim + idx; i < end_idx * dim; i += stride) {
      scalar_t d = x[(old * dim) + (i % dim)] - x[i];
      atomicAdd(&tmp_dist[i / dim], d * d);
    }

    __syncthreads();
    for (ptrdiff_t n = start_idx + idx; n < end_idx; n += stride) {
      dist[n] = min(dist[n], tmp_dist[n]);
    }
  }
}

at::Tensor fps_cuda(at::Tensor x, at::Tensor batch, float ratio, bool random) {
  auto batch_sizes = (int64_t *)malloc(sizeof(int64_t));
  cudaMemcpy(batch_sizes, batch[-1].data<int64_t>(), sizeof(int64_t),
             cudaMemcpyDeviceToHost);
  auto batch_size = batch_sizes[0] + 1;

  auto deg = degree(batch, batch_size);
  auto cum_deg = at::cat({at::zeros(1, deg.options()), deg.cumsum(0)}, 0);
  auto k = (deg.toType(at::kFloat) * ratio).round().toType(at::kLong);
  auto cum_k = at::cat({at::zeros(1, k.options()), k.cumsum(0)}, 0);

  at::Tensor start;
  if (random) {
    start = at::rand(batch_size, x.options());
    start = (start * deg.toType(at::kFloat)).toType(at::kLong);
  } else {
    start = at::zeros(batch_size, k.options());
  }

  auto dist = at::full(x.size(0), 1e38, x.options());
  auto tmp_dist = at::empty(x.size(0), x.options());

  auto k_sum = (int64_t *)malloc(sizeof(int64_t));
  cudaMemcpy(k_sum, cum_k[-1].data<int64_t>(), sizeof(int64_t),
             cudaMemcpyDeviceToHost);
  auto out = at::empty(k_sum[0], k.options());

  AT_DISPATCH_FLOATING_TYPES(x.type(), "fps_kernel", [&] {
    fps_kernel<scalar_t><<<batch_size, THREADS>>>(
        x.data<scalar_t>(), cum_deg.data<int64_t>(), cum_k.data<int64_t>(),
        start.data<int64_t>(), dist.data<scalar_t>(), tmp_dist.data<scalar_t>(),
        out.data<int64_t>(), x.size(1));
  });

  return dist;
}

// at::Tensor ifp_cuda(at::Tensor x, at::Tensor batch, float ratio) {
//   AT_DISPATCH_FLOATING_TYPES(x.type(), "ifp_kernel", [&] {
//     ifp_kernel<scalar_t><<<BLOCKS(x.numel()), THREADS>>>(
//         x.data<scalar_t>(), batch.data<int64_t>(), ratio, x.numel());
//   });

//   return x;
// }

// __global__ void ifps_kernel() {}

// // x: [N, F]
// // count: [B]
// // batch: [N]
// // tmp min distances: [N]
// // start node idx

// // we parallelize over n times f
// // parallelization over n times f: We can compute distances over atomicAdd
// // each block corresponds to a batch

// __global__ void farthestpointsamplingKernel(int b, int n, int m,
//                                             const float *__restrict__
//                                             dataset, float *__restrict__
//                                             temp, int *__restrict__ idxs) {
//   // dataset: [N*3] entries
//   // b: batch-size
//   // n: number of nodes
//   // m: number of sample points

//   if (m <= 0)
//     return;
//   const int BlockSize = 512;
//   __shared__ float dists[BlockSize];
//   __shared__ int dists_i[BlockSize];
//   const int BufferSize = 3072;
//   __shared__ float buf[BufferSize * 3];
//   for (int i = blockIdx.x; i < b; i += gridDim.x) { // iterate over all
//   batches?
//     int old = 0;
//     if (threadIdx.x == 0)
//       idxs[i * m + 0] = old;
//     for (int j = threadIdx.x; j < n; j += blockDim.x) { // iterate over all n
//       temp[blockIdx.x * n + j] = 1e38;
//     }
//     for (int j = threadIdx.x; j < min(BufferSize, n) * 3; j += blockDim.x) {
//       buf[j] = dataset[i * n * 3 + j];
//     }
//     __syncthreads();
//     for (int j = 1; j < m; j++) {
//       int besti = 0;
//       float best = -1;
//       float x1 = dataset[i * n * 3 + old * 3 + 0];
//       float y1 = dataset[i * n * 3 + old * 3 + 1];
//       float z1 = dataset[i * n * 3 + old * 3 + 2];
//       for (int k = threadIdx.x; k < n; k += blockDim.x) {
//         float td = temp[blockIdx.x * n + k];
//         float x2, y2, z2;
//         if (k < BufferSize) {
//           x2 = buf[k * 3 + 0];
//           y2 = buf[k * 3 + 1];
//           z2 = buf[k * 3 + 2];
//         } else {
//           x2 = dataset[i * n * 3 + k * 3 + 0];
//           y2 = dataset[i * n * 3 + k * 3 + 1];
//           z2 = dataset[i * n * 3 + k * 3 + 2];
//         }
//         float d = (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1) +
//                   (z2 - z1) * (z2 - z1);
//         float d2 = min(d, td);
//         if (d2 != td)
//           temp[blockIdx.x * n + k] = d2;
//         if (d2 > best) {
//           best = d2;
//           besti = k;
//         }
//       }
//       dists[threadIdx.x] = best;
//       dists_i[threadIdx.x] = besti;
//       for (int u = 0; (1 << u) < blockDim.x; u++) {
//         __syncthreads();
//         if (threadIdx.x < (blockDim.x >> (u + 1))) {
//           int i1 = (threadIdx.x * 2) << u;
//           int i2 = (threadIdx.x * 2 + 1) << u;
//           if (dists[i1] < dists[i2]) {
//             dists[i1] = dists[i2];
//             dists_i[i1] = dists_i[i2];
//           }
//         }
//       }
//       __syncthreads();
//       old = dists_i[0];
//       if (threadIdx.x == 0)
//         idxs[i * m + j] = old;
//     }
//   }
// }
