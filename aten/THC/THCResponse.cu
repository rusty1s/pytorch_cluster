#include "common.cuh"

__global__ void responseKernel(int64_t *color, int64_t *prop, int64_t *row, int64_t *col,
                              int64_t *deg, int64_t *cumDeg, ptrdiff_t nNodes) {
  KERNEL_LOOP(i, nNodes) {
    if (color[i] != -2) continue; // Only visit red nodes.
    ptrdiff_t c; int64_t neighborColor, minValue;
    bool isDead = true;
    for (ptrdiff_t e = cumDeg[i] - deg[i]; e < cumDeg[i]; e++) {
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
                        THCudaLongTensor *row, THCudaLongTensor *col, THCudaLongTensor *deg,
                        THCudaLongTensor *cumDeg) {
  ptrdiff_t nNodes = THCudaLongTensor_nElement(state, color);
  int64_t *colorData = THCudaLongTensor_data(state, color);
  int64_t *propData = THCudaLongTensor_data(state, prop);
  int64_t *rowData = THCudaLongTensor_data(state, row);
  int64_t *colData = THCudaLongTensor_data(state, col);
  int64_t *degData = THCudaLongTensor_data(state, deg);
  int64_t *cumDegData = THCudaLongTensor_data(state, cumDeg);
  KERNEL_RUN(responseKernel, nNodes, colorData, propData, rowData, colData, degData, cumDegData);
}
