#include "THCGreedy.h"

#include "common.cuh"
#include "THCDegree.cu"
#include "THCColor.cu"

__global__ void proposeKernel(int64_t *tensor, int64_t *color, int64_t *row, int64_t *col,
                               int64_t *deg, int64_t *cumDeg, ptrdiff_t nNodes) {
  KERNEL_LOOP(i, nNodes) {
    if (color[i] != -1) continue; // Only visit blue nodes.
    ptrdiff_t c;
    for (ptrdiff_t e = cumDeg[i]; e < cumDeg[i] + deg[i]; e++) {
      c = col[e];
      if (color[c] == -2) { tensor[i] = c; break; }  // Propose to first red node.
    }
    if (tensor[i] < 0) color[i] = i;  // Mark node as dead.
  }
}

void THCGreedy_propose(THCState *state, THCudaLongTensor *tensor, THCudaLongTensor *color,
                        THCudaLongTensor *row, THCudaLongTensor *col, THCudaLongTensor *deg,
                        THCudaLongTensor *cumDeg) {
  ptrdiff_t nNodes = THCudaLongTensor_nElement(state, color);
  int64_t *tensorData = THCudaLongTensor_data(state, tensor);
  int64_t *colorData = THCudaLongTensor_data(state, color);
  int64_t *rowData = THCudaLongTensor_data(state, row);
  int64_t *colData = THCudaLongTensor_data(state, col);
  int64_t *degData = THCudaLongTensor_data(state, deg);
  int64_t *cumDegData = THCudaLongTensor_data(state, cumDeg);
  KERNEL_RUN(proposeKernel, nNodes, tensorData, colorData, rowData, colData, degData, cumDegData);
}

void THCGreedy(THCState *state, THCudaLongTensor *cluster, THCudaLongTensor *row,
               THCudaLongTensor *col) {
  THCAssertSameGPU(THCudaLongTensor_checkGPU(state, 4, cluster, row, col));

  int nNodes = THCudaLongTensor_nElement(state, cluster);

  THCudaLongTensor_fill(state, cluster, -1);
  THCudaLongTensor *prop = THCudaLongTensor_newClone(state, cluster);

  THCudaLongTensor *deg = THCudaLongTensor_newWithSize1d(state, nNodes);
  THCDegree(state, deg, row);

  THCudaLongTensor *cumDeg = THCudaLongTensor_newWithSize1d(state, nNodes);
  THCudaLongTensor_cumsum(state, cumDeg, deg, 0);

  THCGreedy_assignColor(state, cluster);
  THCGreedy_propose(state, prop, cluster, row, col, deg, cumDeg);

  /* while(!THCGreedy_assignColor(state, cluster)) { */
  /*   printf("DRIN"); */
  // call propose step
  // call response step
  /* }; */
  THCudaLongTensor_free(state, prop);
  THCudaLongTensor_free(state, deg);
  THCudaLongTensor_free(state, cumDeg);
}

#include "generic/THCGreedy.cu"
#include "THC/THCGenerateAllTypes.h"
