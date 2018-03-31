#include <curand.h>
#include <curand_kernel.h>

#include "common.cuh"

__global__ void assignColorKernel(int64_t *color, curandStateMtgp32 *state, uint8_t *done,
                                  ptrdiff_t nNodes) {
  KERNEL_LOOP(i, nNodes) {
    if (color[i] < 0) {
      color[i] = (curand_uniform(&state[0]) < 0.53406) - 2;  // blue = -1, red = -2
      *done = 0;
    }
  }
}

int THCGreedy_assignColor(THCState *state, THCudaLongTensor *color) {
  int64_t *colorData = THCudaLongTensor_data(state, color);
  ptrdiff_t nNodes = THCudaLongTensor_nElement(state, color);
  uint8_t* d_done; cudaMalloc(&d_done, sizeof(uint8_t)); cudaMemset(d_done, 1, sizeof(uint8_t));
  KERNEL_RUN(assignColorKernel, nNodes, colorData, THCRandom_generatorStates(state), d_done);
  uint8_t done; cudaMemcpy(&done, d_done, sizeof(uint8_t), cudaMemcpyDeviceToHost);
  cudaFree(d_done);
  return done;
}
