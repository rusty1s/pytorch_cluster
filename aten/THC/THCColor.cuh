#ifndef THC_COLOR_INC
#define THC_COLOR_INC

#include "common.cuh"

#include <curand.h>
#include <curand_kernel.h>

#define BLUE_PROBABILITY 0.53406

__global__ void colorKernel(int64_t *self, curandStateMtgp32 *state, uint8_t *done,
                                  ptrdiff_t nNodes) {
  KERNEL_LOOP(i, nNodes) {
    if (self[i] < 0) {
      self[i] = (curand_uniform(&state[0]) < BLUE_PROBABILITY) - 2;  // blue = -1, red = -2
      *done = 0;
    }
  }
}

int THCTensor_color(THCState *state, THCudaLongTensor *self) {
  uint8_t* d_done;
  cudaMalloc(&d_done, sizeof(uint8_t));
  cudaMemset(d_done, 1, sizeof(uint8_t));

  ptrdiff_t nNodes = THCudaLongTensor_nElement(state, self);
  int64_t *selfData = THCudaLongTensor_data(state, self);

  KERNEL_RUN(colorKernel, nNodes, selfData, THCRandom_generatorStates(state), d_done);

  uint8_t done;
  cudaMemcpy(&done, d_done, sizeof(uint8_t), cudaMemcpyDeviceToHost);
  cudaFree(d_done);

  return done;
}

#endif  // THC_COLOR_INC
