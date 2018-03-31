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

  THCudaLongTensor *degree = THCudaLongTensor_newWithSize1d(state, nNodes);
  THCudaLongTensor_degree(state, degree, row);

  THCudaLongTensor *cumDegree = THCudaLongTensor_newWithSize1d(state, nNodes);
  THCudaLongTensor_cumsum(state, cumDegree, degree, 0);

  while(!THCGreedy_assignColor(state, cluster)) {
    THCGreedy_propose(state, cluster, prop, row, col, degree, cumDegree);
    THCGreedy_response(state, cluster, prop, row, col, degree, cumDegree);
  };

  THCudaLongTensor_free(state, prop);
  THCudaLongTensor_free(state, degree);
  THCudaLongTensor_free(state, cumDegree);
}

#include "generic/THCGreedy.cu"
#include "THC/THCGenerateAllTypes.h"
