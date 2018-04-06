#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/THCGraclus.h"
#else

void THCTensor_(graclus)(THCState *state, THCudaLongTensor *self, THCudaLongTensor *row,
                         THCudaLongTensor *col, THCTensor *weight);

#endif  // THC_GENERIC_FILE
