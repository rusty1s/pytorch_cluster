#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <stdint.h>
#include <vector>

#define THREADS 1024

// Original by Qi. et al (https://github.com/charlesq34/pointnet2)

template <typename scalar_t>
__global__ void query_radius_cuda_kernel(
    const int64_t* __restrict__ batch_slices,
    const int64_t* __restrict__ query_batch_slices,
    const scalar_t* __restrict__ pos,
    const scalar_t* __restrict__ query_pos,
    const scalar_t radius,
    const int64_t max_num_neighbors,
    const bool include_self,
    int64_t* idx_output,
    int64_t* cnt_output)
{
    const int64_t batch_index = blockIdx.x;
    const int64_t index = threadIdx.x;
    const int64_t stride = blockDim.x;

    const int64_t batch_start = batch_slices[2*batch_index];
    const int64_t query_batch_start = query_batch_slices[2*batch_index];
    const int64_t batch_end = batch_slices[2*batch_index+1];
    const int64_t query_batch_end = query_batch_slices[2*batch_index+1];

    const int64_t batch_size = batch_end - batch_start + 1;
    const int64_t query_batch_size = query_batch_end - query_batch_start + 1;

    pos += batch_start * 3;
    query_pos += query_batch_start * 3;
    idx_output += query_batch_start * max_num_neighbors;
    cnt_output += query_batch_start;


    for (int64_t j = index; j < query_batch_size; j+=stride){

        int64_t cnt = 0;
        scalar_t x2=query_pos[j*3+0];
        scalar_t y2=query_pos[j*3+1];
        scalar_t z2=query_pos[j*3+2];

        // dummy outputs initialisation with value -1
        if (cnt==0) {
            for (int64_t l = 0;l < max_num_neighbors; l++)
                idx_output[j*max_num_neighbors+l] = -1;
        }

        for (int64_t k = 0; k < batch_size; k++) {
            if (cnt == max_num_neighbors)
                break;

            scalar_t x1=pos[k*3+0];
            scalar_t y1=pos[k*3+1];
            scalar_t z1=pos[k*3+2];

            scalar_t d=(x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)+(z2-z1)*(z2-z1);

            if (d <= radius && (d > 0 || include_self)) {
                idx_output[j * max_num_neighbors + cnt] = batch_start + k;
                cnt+=1;
            }
        }
        cnt_output[j] = cnt;
    }
}

template <typename scalar_t>
__global__ void query_knn_cuda_kernel(
    const int64_t* __restrict__ batch_slices,
    const int64_t* __restrict__ query_batch_slices,
    const scalar_t* __restrict__ pos,
    const scalar_t* __restrict__ query_pos,
    const int64_t num_neighbors,
    const bool include_self,
    scalar_t* tmp_dists,
    int64_t* idx_output){

    const int64_t batch_index = blockIdx.x;
    const int64_t index = threadIdx.x;
    const int64_t stride = blockDim.x;

    const int64_t batch_start = batch_slices[2*batch_index];
    const int64_t query_batch_start = query_batch_slices[2*batch_index];
    const int64_t batch_end = batch_slices[2*batch_index+1];
    const int64_t query_batch_end = query_batch_slices[2*batch_index+1];

    const int64_t batch_size = batch_end - batch_start + 1;
    const int64_t query_batch_size = query_batch_end - query_batch_start + 1;

    pos += batch_start * 3;
    query_pos += query_batch_start * 3;
    idx_output += query_batch_start * num_neighbors;
    tmp_dists += query_batch_start * num_neighbors;

    for (int64_t j = index; j < query_batch_size; j += stride){
        scalar_t x2=query_pos[j*3+0];
        scalar_t y2=query_pos[j*3+1];
        scalar_t z2=query_pos[j*3+2];
        // reset to dummy values

        for (int64_t l = 0; l < num_neighbors; l++){
            idx_output[j * num_neighbors + l] = -1;
            tmp_dists[j * num_neighbors + l] = 2147483647;
        }

        for (int64_t k = 0; k < batch_size; k++) {
            scalar_t x1=pos[k*3+0];
            scalar_t y1=pos[k*3+1];
            scalar_t z1=pos[k*3+2];

            scalar_t d=(x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)+(z2-z1)*(z2-z1);

            if (d > 0 || include_self){
                for (int64_t i = 0; i < num_neighbors; i++){
                    if (tmp_dists[j * num_neighbors + i] > d){
                        for (int64_t i2 = num_neighbors-1; i2 > i; i2--){
                            tmp_dists[j * num_neighbors + i2] = tmp_dists[j * num_neighbors + i2 - 1];
                            idx_output[j * num_neighbors + i2] = idx_output[j * num_neighbors + i2 - 1];
                        }
                        tmp_dists[j * num_neighbors + i] = d;
                        idx_output[j * num_neighbors + i] = batch_start + k;
                        break;
                    }
                }
            }
        }
    }
}

template <typename scalar_t>
__global__ void farthest_point_sampling_kernel(
    const int64_t* __restrict__ batch_slices,
    const scalar_t* __restrict__ pos,
    const int64_t num_sample,
    const int64_t* __restrict__ start_points,
    scalar_t* tmp_dists,
    int64_t* idx_output){

    const int64_t batch_index = blockIdx.x;
    const int64_t index = threadIdx.x;
    const int64_t stride = blockDim.x;

    const int64_t batch_start = batch_slices[2*batch_index];
    const int64_t batch_end = batch_slices[2*batch_index+1];
    const int64_t batch_size = batch_end - batch_start + 1;

  __shared__ scalar_t dists[THREADS];
  __shared__ int64_t dists_i[THREADS];

    pos += batch_start * 3;
    idx_output += num_sample * batch_index;
    tmp_dists += batch_start;

    int64_t old = start_points[batch_index];

    // explicitly handle the case where less than num_sample points are available to sample from
    if (index == 0){
        idx_output[0] = batch_start + old;

        if (batch_size < num_sample){
            for (int64_t i = 0; i < batch_size; i++){
                idx_output[i] = batch_start + i;
            }
            for (int64_t i = batch_size; i < num_sample; i++){
                idx_output[i] = -1;
            }
        }
     }
    if (batch_size < num_sample){
        return;
    }

    // initialise temporary distances as big as possible
    for (int64_t j = index; j < batch_size; j+=stride){
        tmp_dists[j] = 2147483647;
    }

    __syncthreads();
    for (int64_t i = 1; i < num_sample; i++){
        int64_t besti = -1;
        scalar_t best = -1;

        // compute distance from last point to all others and update using the minimum of already computed distances
        for (int64_t j = index; j < batch_size; j+= stride){
            scalar_t td = tmp_dists[j];
            scalar_t x1 = pos[old * 3 + 0];
            scalar_t y1 = pos[old * 3 + 1];
            scalar_t z1 = pos[old * 3 + 2];

            scalar_t x2 = pos[j * 3 + 0];
            scalar_t y2 = pos[j * 3 + 1];
            scalar_t z2 = pos[j * 3 + 2];

            scalar_t d=(x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)+(z2-z1)*(z2-z1);
            scalar_t d2  = min(d, tmp_dists[j]);
            if (td != d2){
                tmp_dists[j] = d2;
            }

            if (tmp_dists[j] > best){
              best = tmp_dists[j];
              besti = j;
            }
        }

        // sort best indices
        dists[index] = best;
        dists_i[index] = besti;

        __syncthreads();
        // get the maximum distances (by merging)
        for (int64_t u = 0; (1<<u) < blockDim.x ; u++){
            __syncthreads();
            if (index < (blockDim.x >> (u+1))){
                int64_t i1 = (index*2)<<u;
                int64_t i2 = (index*2+1)<<u;
                if (dists[i1] < dists[i2]){
                    dists[i1] = dists[i2];
                    dists_i[i1] = dists_i[i2];
                }
            }
        }

        __syncthreads();

        if (dists[0] == 0){
            break;
        }
        // thread 0 collects in output
        old = dists_i[0];
        if (index == 0){
            idx_output[i] = batch_start + old;
        }
    }

}


std::vector<at::Tensor> query_radius_cuda(
    int batch_size,
    at::Tensor batch_slices,
    at::Tensor query_batch_slices,
    at::Tensor pos,
    at::Tensor query_pos,
    double radius,
    int max_num_neighbors,
    bool include_self) {

  const auto num_points = query_pos.size(0);

  auto idx_output = at::empty(pos.type().toScalarType(at::kLong), {num_points, max_num_neighbors});
  auto cnt_output = at::empty(pos.type().toScalarType(at::kLong), {num_points});

  AT_DISPATCH_FLOATING_TYPES(pos.type(), "query_radius_cuda_kernel", [&] {
      query_radius_cuda_kernel<scalar_t><<<batch_size, THREADS>>>(
        batch_slices.data<int64_t>(),
        query_batch_slices.data<int64_t>(),
        pos.data<scalar_t>(),
        query_pos.data<scalar_t>(),
        (scalar_t) radius*radius,
        max_num_neighbors,
        include_self,
        idx_output.data<int64_t>(),
        cnt_output.data<int64_t>());
  });


  return {idx_output, cnt_output};
}


std::vector<at::Tensor> query_knn_cuda(
    int batch_size,
    at::Tensor batch_slices,
    at::Tensor query_batch_slices,
    at::Tensor pos,
    at::Tensor query_pos,
    int num_neighbors,
    bool include_self) {

  const auto num_points = query_pos.size(0);

  auto idx_output = at::empty(pos.type().toScalarType(at::kLong), {num_points, num_neighbors});
  auto dists = at::empty(pos.type(), {num_points, num_neighbors});

  AT_DISPATCH_FLOATING_TYPES(pos.type(), "query_knn_cuda_kernel", [&] {
    query_knn_cuda_kernel<scalar_t><<<batch_size, THREADS>>>(
      batch_slices.data<int64_t>(),
      query_batch_slices.data<int64_t>(),
      pos.data<scalar_t>(),
      query_pos.data<scalar_t>(),
      num_neighbors,
      include_self,
      dists.data<scalar_t>(),
      idx_output.data<int64_t>());
  });


  return {idx_output, dists};
}

at::Tensor farthest_point_sampling_cuda(
    int batch_size,
    at::Tensor batch_slices,
    at::Tensor pos,
    int num_sample,
    at::Tensor start_points) {

  auto idx_output = at::empty(pos.type().toScalarType(at::kLong), {batch_size * num_sample});
  auto tmp_dists = at::empty(pos.type(), {pos.size(0)});

  AT_DISPATCH_FLOATING_TYPES(pos.type(), "farthest_point_sampling_kernel", [&] {
    farthest_point_sampling_kernel<scalar_t><<<batch_size, THREADS>>>(
      batch_slices.data<int64_t>(),
      pos.data<scalar_t>(),
      num_sample,
      start_points.data<int64_t>(),
      tmp_dists.data<scalar_t>(),
      idx_output.data<int64_t>());
  });




  return idx_output;
}

