#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/THCCGrid.c"
#else

void THCCGrid_(THCudaLongTensor *cluster, THCTensor *pos, THCTensor *size,
               THCudaLongTensor *count) {
  THCGrid_(state, cluster, pos, size, count);
}

#endif

