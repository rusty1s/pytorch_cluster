#include "common.cuh"

__global__ void proposeKernel(int64_t *color, int64_t *prop, int64_t *row, int64_t *col,
                              int64_t *deg, int64_t *cumDeg, ptrdiff_t nNodes) {
  KERNEL_LOOP(i, nNodes) {
    if (color[i] != -1) continue; // Only visit blue nodes.
    ptrdiff_t c;
    for (ptrdiff_t e = cumDeg[i] - deg[i]; e < cumDeg[i]; e++) {
      c = col[e];
      if (color[c] == -2) {  // Red neighbor found.
        prop[i] = c;  // Propose!
        break;
      }
    }
    if (prop[i] < 0) color[i] = i;  // Mark node as dead.
  }
}

void THCGreedy_propose(THCState *state, THCudaLongTensor *color, THCudaLongTensor *prop,
                       THCudaLongTensor *row, THCudaLongTensor *col, THCudaLongTensor *deg,
                       THCudaLongTensor *cumDeg) {
  ptrdiff_t nNodes = THCudaLongTensor_nElement(state, color);
  int64_t *colorData = THCudaLongTensor_data(state, color);
  int64_t *propData = THCudaLongTensor_data(state, prop);
  int64_t *rowData = THCudaLongTensor_data(state, row);
  int64_t *colData = THCudaLongTensor_data(state, col);
  int64_t *degData = THCudaLongTensor_data(state, deg);
  int64_t *cumDegData = THCudaLongTensor_data(state, cumDeg);
  KERNEL_RUN(proposeKernel, nNodes, colorData, propData, rowData, colData, degData, cumDegData);
}
