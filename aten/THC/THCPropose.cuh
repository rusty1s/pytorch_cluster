#ifndef THC_PROPOSE_INC
#define THC_PROPOSE_INC

#include "common.cuh"

__global__ void proposeKernel(int64_t *color, int64_t *prop, int64_t *row, int64_t *col,
                              int64_t *degree, int64_t *cumDegree, ptrdiff_t nNodes) {
  KERNEL_LOOP(i, nNodes) {
    if (color[i] != -1) { continue; }  // Only visit blue nodes.
    ptrdiff_t c; bool isDead = true;
    for (ptrdiff_t e = cumDegree[i] - degree[i]; e < cumDegree[i]; e++) {
      c = col[e];
      if (isDead && color[c] < 0) { isDead = false; }  // Unmatched neighbor found.
      if (color[c] == -2) { prop[i] = c; break; }  // Propose to first red neighbor.
    }
    if (isDead) { color[i] = i; }  // Mark node as dead.
  }
}

void THCTensor_propose(THCState *state, THCudaLongTensor *color, THCudaLongTensor *prop,
                       THCudaLongTensor *row, THCudaLongTensor *col, THCudaLongTensor *degree,
                       THCudaLongTensor *cumDegree) {
  KERNEL_RUN(proposeKernel, THCudaLongTensor_nElement(state, color),
             THCudaLongTensor_data(state, color), THCudaLongTensor_data(state, prop),
             THCudaLongTensor_data(state, row), THCudaLongTensor_data(state, col),
             THCudaLongTensor_data(state, degree), THCudaLongTensor_data(state, cumDegree));
}

#endif  // THC_PROPOSE_INC
