#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/THCGreedy.h"
#else

void THCGreedy_(THCState *state, THCudaLongTensor *cluster, THCudaLongTensor *row,
                THCudaLongTensor *col, THCudaLongTensor *deg, THCTensor *weight);

#endif  // THC_GENERIC_FILE
