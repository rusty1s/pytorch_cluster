#include "THCGreedy.h"

#include "THCColor.cu"

void THCGreedy(THCState *state, THCudaLongTensor *cluster, THCudaLongTensor *row,
               THCudaLongTensor *col, THCudaLongTensor *deg) {
  THCAssertSameGPU(THCudaLongTensor_checkGPU(state, 4, cluster, row, col, deg));

  THCudaLongTensor_fill(state, cluster, -1);
  THCGreedy_assignColor(state, cluster);
  /* while(!THCGreedy_assignColor(state, cluster)) { */
  /*   printf("DRIN"); */
  /* }; */

  // Fill cluster with -1
  // assign color to clusters < 0 (return done)
  // Generate proposal vector with length of nodes (init?)
  // call propose step
  // call response step
}

#include "generic/THCGreedy.cu"
#include "THC/THCGenerateAllTypes.h"
