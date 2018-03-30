#include "THCGreedy.h"

void THCGreedy_cluster(THCState *state,
                       THCudaLongTensor *cluster,
                       THCudaLongTensor *row,
                       THCudaLongTensor *col,
                       THCudaLongTensor *deg) {
  printf("THCGreedy_cluster drin");
  // Fill cluster with -1
  // assign color to clusters < 0 (return done)
  // Generate proposal vector with length of nodes (init?)
  // call propose step
  // call response step
}

#include "generic/THCGreedy.cu"
#include "THC/THCGenerateAllTypes.h"
