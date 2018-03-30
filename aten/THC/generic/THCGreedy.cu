#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/THCGreedy.cu"
#else

void THCGreedy_(THCState *state, THCudaLongTensor *cluster, THCudaLongTensor *row,
                THCudaLongTensor *col, THCudaLongTensor *deg, THCTensor *weight) {
  printf("THCGreedy dynamic drin");
}

#endif  // THC_GENERIC_FILE


