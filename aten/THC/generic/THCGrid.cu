#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/THCGrid.cu"
#else

void THCTensor_(grid)(THCState *state, THCudaLongTensor *self, THCTensor *pos, THCTensor *size,
                      THCudaLongTensor *count) {
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 4, self, pos, size, count));

  ptrdiff_t nNodes = THCudaLongTensor_nElement(state, self);

  int64_t *selfData = THCudaLongTensor_data(state, self);
  TensorInfo<real> posInfo = THCTensor_(getTensorInfo)(state, pos);
  real *sizeData = THCTensor_(data)(state, size);
  int64_t *countData = THCudaLongTensor_data(state, count);

  KERNEL_REAL_RUN(gridKernel, nNodes, selfData, posInfo, sizeData, countData);
}

#endif  // THC_GENERIC_FILE
