#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/serial_cuda.c"
#else

void cluster_(serial)(THCudaLongTensor *output, THCudaLongTensor *row, THCudaLongTensor *col, THCudaLongTensor *degree, THCTensor *weight) {
  cluster_kernel_(serial)(state, output, row, col, degree, weight);
}

#endif

