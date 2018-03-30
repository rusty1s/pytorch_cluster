#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/THCCGreedy.c"
#else

void THCCGreedy_(THCudaLongTensor *cluster, THCudaLongTensor *row, THCudaLongTensor *col,
                 THCudaLongTensor *deg, THCTensor *weight) {
  THCGreedy_(state, cluster, row, col, deg, weight);
}

#endif
