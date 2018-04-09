#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/THCCGraclus.c"
#else

void THCCTensor_(graclus)(THCudaLongTensor *self, THCudaLongTensor *row, THCudaLongTensor *col,
                         THCTensor *weight) {
  THCTensor_(graclus)(state, self, row, col, weight);
}

#endif  // THC_GENERIC_FILE
