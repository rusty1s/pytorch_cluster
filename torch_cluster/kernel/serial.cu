#include <THC.h>
#include "THCTensorRandom.h"

#include "serial.h"

#include <curand.h>
#include <curand_kernel.h>
#include "common.cuh"

#define cluster_(NAME) TH_CONCAT_4(cluster_, NAME, _kernel_, Real)
#define thc_(NAME) TH_CONCAT_4(thc_, NAME, _, Real)

__global__ void assignColorKernel(curandStateMtgp32 *state, int64_t *color, const int n, uint8_t *done) {
  KERNEL_LOOP(i, n) {
    if (color[i] < 0) {
      color[i] = 0; //(int64_t) (curand_uniform(&state[blockIdx.x]) <= 0.53406) - 2;
      *done = 0;
    }

  }
}

int assignColor(THCState *state, THCudaLongTensor *color) {
  curandStateMtgp32 *gen_states = THCRandom_generatorStates(state);
  int64_t *colorVec = THCudaLongTensor_data(state, color);
  const int n = THCudaLongTensor_nElement(state, color);
  uint8_t done; uint8_t* d_done; cudaMalloc(&d_done, sizeof(uint8_t)); cudaMemset(d_done, 1, sizeof(uint8_t)); // *(done) = (int) 1;
  assignColorKernel<<<GET_BLOCKS(n), NUM_THREADS, 0, THCState_getCurrentStream(state)>>>(gen_states, colorVec, n, d_done);
  cudaMemcpy(&done, d_done, sizeof(uint8_t), cudaMemcpyDeviceToHost); cudaFree(d_done);
  return done;
}


/* GENERATE_KERNEL1(generate_bernoulli, double, double p, double, curand_uniform_double, x <= p) */

/* #define GENERATE_KERNEL1(NAME, T, ARG1, CURAND_T, CURAND_FUNC, TRANSFORM)      \ */
/* __global__ void NAME(curandStateMtgp32 *state, int size, T *result, ARG1)      \ */
/* {                                                                              \ */
/*   int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;                             \ */
/*   int rounded_size = THCCeilDiv(size, BLOCK_SIZE) * BLOCK_SIZE;                \ */
/*   for (int i = idx; i < rounded_size; i += BLOCK_SIZE * MAX_NUM_BLOCKS) {      \ */
/*     CURAND_T x = CURAND_FUNC(&state[blockIdx.x]);                              \ */
/*     if (i < size) {                                                            \ */
/*       T y = TRANSFORM;                                                         \ */
/*       result[i] = y;                                                           \ */
/*     }                                                                          \ */
/*   }                                                                            \ */
/* } */

void cluster_serial_kernel(THCState *state, THCudaLongTensor *output, THCudaLongTensor *row, THCudaLongTensor *col, THCudaLongTensor *degree) {
}

#include "generic/serial.cu"
#include "THCGenerateFloatType.h"
#include "generic/serial.cu"
#include "THCGenerateDoubleType.h"
#include "generic/serial.cu"
#include "THCGenerateByteType.h"
#include "generic/serial.cu"
#include "THCGenerateCharType.h"
#include "generic/serial.cu"
#include "THCGenerateShortType.h"
#include "generic/serial.cu"
#include "THCGenerateIntType.h"
#include "generic/serial.cu"
#include "THCGenerateLongType.h"
