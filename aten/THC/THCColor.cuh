#ifndef THC_COLOR_INC
#define THC_COLOR_INC

#include "common.cuh"

#define BLUE_PROB 0.53406

__device__ int d_done;
__global__ void initDoneKernel() { d_done = 1; }

__global__ void colorKernel(int64_t *self, uint8_t *bernoulli, ptrdiff_t nNodes) {
  KERNEL_LOOP(i, nNodes) {
    if (self[i] < 0) {
      self[i] = bernoulli[i] - 2;
      d_done = 0;
    }
  }
}

int THCudaLongTensor_color(THCState *state, THCudaLongTensor *self) {
  initDoneKernel<<<1, 1>>>();

  ptrdiff_t nNodes = THCudaLongTensor_nElement(state, self);

  THCudaByteTensor *bernoulli = THCudaByteTensor_newWithSize1d(state, nNodes);
  THCudaByteTensor_bernoulli(state, bernoulli, BLUE_PROB);

  int64_t *selfData = THCudaLongTensor_data(state, self);
  uint8_t *bernoulliData = THCudaByteTensor_data(state, bernoulli);

  KERNEL_RUN(colorKernel, nNodes, selfData, bernoulliData);

  int done; cudaMemcpyFromSymbol(&done, d_done, sizeof(done), 0, cudaMemcpyDeviceToHost);
  return done;
}

#endif  // THC_COLOR_INC
