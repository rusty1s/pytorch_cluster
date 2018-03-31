#include "THCGreedy.h"

#include "common.cuh"
#include "THCDegree.cu"
#include "THCColor.cu"
#include "THCPropose.cu"
#include "THCResponse.cu"

void THCGreedy(THCState *state, THCudaLongTensor *cluster, THCudaLongTensor *row,
               THCudaLongTensor *col) {
  THCAssertSameGPU(THCudaLongTensor_checkGPU(state, 3, cluster, row, col));

  int nNodes = THCudaLongTensor_nElement(state, cluster);

  THCudaLongTensor_fill(state, cluster, -1);
  THCudaLongTensor *prop = THCudaLongTensor_newClone(state, cluster);

  THCudaLongTensor *deg = THCudaLongTensor_newWithSize1d(state, nNodes);
  THCDegree(state, deg, row);

  THCudaLongTensor *cumDeg = THCudaLongTensor_newWithSize1d(state, nNodes);
  THCudaLongTensor_cumsum(state, cumDeg, deg, 0);

  while(!THCGreedy_assignColor(state, cluster)) {
    THCGreedy_propose(state, cluster, prop, row, col, deg, cumDeg);
    THCGreedy_response(state, cluster, prop, row, col, deg, cumDeg);
  };

  THCudaLongTensor_free(state, prop);
  THCudaLongTensor_free(state, deg);
  THCudaLongTensor_free(state, cumDeg);
}

#include "generic/THCGreedy.cu"
#include "THC/THCGenerateAllTypes.h"
