#include "common.cuh"

__global__ void responseKernel(int64_t *color, int64_t *prop, int64_t *row, int64_t *col,
                               int64_t *degree, int64_t *cumDegree, ptrdiff_t nNodes) {
  KERNEL_LOOP(i, nNodes) {
    if (color[i] != -2) continue; // Only visit red nodes.
    ptrdiff_t c; int64_t neighborColor, minValue;
    bool isDead = true;
    for (ptrdiff_t e = cumDegree[i] - degree[i]; e < cumDegree[i]; e++) {
      c = col[e];
      neighborColor = color[c];
      if (neighborColor == -1 && prop[c] == i) {  // Blue neighbor found which proposed to node i.
        minValue = min(i, c);
        color[i] = minValue;
        color[c] = minValue;
        break;
      }
      if (neighborColor < 0) isDead = false;
    }
    if (isDead && color[i] < 0) color[i] = i;  // Mark node as dead.
  }
}

void THCGreedy_response(THCState *state, THCudaLongTensor *color, THCudaLongTensor *prop,
                        THCudaLongTensor *row, THCudaLongTensor *col, THCudaLongTensor *degree,
                        THCudaLongTensor *cumDegree) {
  KERNEL_RUN(responseKernel, THCudaLongTensor_nElement(state, color),
             THCudaLongTensor_data(state, color), THCudaLongTensor_data(state, prop),
             THCudaLongTensor_data(state, row), THCudaLongTensor_data(state, col),
             THCudaLongTensor_data(state, degree), THCudaLongTensor_data(state, cumDegree))
}
