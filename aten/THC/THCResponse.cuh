#ifndef THC_RESPONSE_INC
#define THC_RESPONSE_INC

#include "common.cuh"

__global__ void responseKernel(int64_t *color, int64_t *prop, int64_t *row, int64_t *col,
                               int64_t *degree, int64_t *cumDegree, ptrdiff_t nNodes) {
  KERNEL_LOOP(i, nNodes) {
    if (color[i] != -2) { continue; }  // Only visit red nodes.
    ptrdiff_t c; bool isDead = true;
    for (ptrdiff_t e = cumDegree[i] - degree[i]; e < cumDegree[i]; e++) {
      c = col[e];
      if (isDead && color[c] < 0) { isDead = false; }  // Unmatched neighbor found.
      if (color[c] == -1 && prop[c] == i) {  // Match first blue neighbor who proposed to i.
        color[i] = min(i, c);
        color[c] = min(i, c);
        break;
      }
    }
    if (isDead) { color[i] = i; }  // Mark node as dead.
  }
}

template<typename T>
__global__ void weightedResponseKernel(int64_t *color, int64_t *prop, int64_t *row, int64_t *col,
                                       T *weight, int64_t *degree, int64_t *cumDegree,
                                       ptrdiff_t nNodes) {
  KERNEL_LOOP(i, nNodes) {
    if (color[i] != -2) { continue; }  // Only visit red nodes.
    ptrdiff_t c; bool isDead = true;
    T maxWeight = ScalarConvert<int, T>::to(0), tmp;
    ptrdiff_t matchedValue = -1;
    for (ptrdiff_t e = cumDegree[i] - degree[i]; e < cumDegree[i]; e++) {
      c = col[e];
      tmp = weight[e];
      if (isDead && color[c] < 0) { isDead = false; }  // Unmatched neighbor found.
      // Find maximum weighted blue neighbor, who proposed to i.
      if (color[c] == -1 && prop[c] == i && THCNumerics<T>::gt(tmp, maxWeight)) {
        matchedValue = c;
        maxWeight = tmp;
      }
    }
    if (matchedValue >= 0) {  // Match neighbors.
      color[i] = min(i, matchedValue);
      color[matchedValue] = min(i, matchedValue);
    }
    if (isDead) { color[i] = i; }  // Mark node as dead.
  }
}

void THCTensor_response(THCState *state, THCudaLongTensor *color, THCudaLongTensor *prop,
                        THCudaLongTensor *row, THCudaLongTensor *col, THCudaLongTensor *degree,
                        THCudaLongTensor *cumDegree) {
  KERNEL_RUN(responseKernel, THCudaLongTensor_nElement(state, color),
             THCudaLongTensor_data(state, color), THCudaLongTensor_data(state, prop),
             THCudaLongTensor_data(state, row), THCudaLongTensor_data(state, col),
             THCudaLongTensor_data(state, degree), THCudaLongTensor_data(state, cumDegree));
}

#include "generic/THCResponse.cuh"
#include "THC/THCGenerateAllTypes.h"

#endif  // THC_RESPONSE_INC
