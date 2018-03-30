#include "THCGreedy.h"

void THCGreedy(THCState *state, THCudaLongTensor *cluster, THCudaLongTensor *row,
               THCudaLongTensor *col, THCudaLongTensor *deg) {
  printf("THCGreedy drin");
  // Fill cluster with -1
  // assign color to clusters < 0 (return done)
  // Generate proposal vector with length of nodes (init?)
  // call propose step
  // call response step
}

#include "generic/THCGreedy.cu"
#include "THC/THCGenerateAllTypes.h"
