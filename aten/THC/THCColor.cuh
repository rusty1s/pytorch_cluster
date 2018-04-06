#ifndef THC_COLOR_INC
#define THC_COLOR_INC

#include "common.cuh"

#define BLUE_PROBABILITY 0.53406

__global__ void colorKernel(int64_t *self, int64_t *bernoulli, uint8_t *done, ptrdiff_t nNodes) {
  KERNEL_LOOP(i, nNodes) {
    if (self[i] < 0) {
      self[i] = bernoulli[i] - 2;
      *done = 0;
    }
  }
}

int THCTensor_color(THCState *state, THCudaLongTensor *self) {
  ptrdiff_t nNodes = THCudaLongTensor_nElement(state, self);
  THCudaLongTensor *bernoulli = THCudaLongTensor_newWithSize1d(state, nNodes);
  THCudaLongTensor_bernoulli(state, bernoulli, BLUE_PROBABILITY);

  int64_t *selfData = THCudaLongTensor_data(state, self);
  int64_t *bernoulliData = THCudaLongTensor_data(state, bernoulli);

  uint8_t* d_done;
  cudaMalloc(&d_done, sizeof(uint8_t));
  cudaMemset(d_done, 1, sizeof(uint8_t));

  KERNEL_RUN(colorKernel, nNodes, selfData, bernoulliData, d_done);

  uint8_t done;
  cudaMemcpy(&done, d_done, sizeof(uint8_t), cudaMemcpyDeviceToHost);
  cudaFree(d_done);

  return done;
}

#endif  // THC_COLOR_INC
