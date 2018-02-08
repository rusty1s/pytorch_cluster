#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/cuda.c"
#else

int64_t cluster_(grid)(THCudaLongTensor *output, THCTensor *position, THCTensor *size, THCTensor *maxPosition) {
  return cluster_kernel_(grid)(state, output, position, size, maxPosition);
}

#endif
