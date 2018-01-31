#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/cuda.c"
#else

void cluster_(grid)(int C, THCudaLongTensor *output, THCTensor *position, THCTensor *size, THCudaLongTensor *count) {
  return cluster_kernel_(grid)(state, C, output, position, size, count);
}

#endif
