#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/THCGrid.cu"
#else

void THCGrid_(THCState *state, THCudaLongTensor *cluster, THCTensor *pos, THCTensor *size,
              THCudaLongTensor *count) {
  THC_assertSameGPU(state, 4, cluster, pos, size, count);

  int64_t *clusterData = THCudaLongTensor_data(state, cluster);
  TensorInfo<real> posInfo = THCTensor_(getTensorInfo)(state, pos);
  real *sizeData = THCTensor_(data)(state, size);
  int64_t *countData = THCudaLongTensor_data(state, count);

  const int nNodes = THCudaLongTensor_nElement(state, cluster);
  KERNEL_RUN(gridKernel, nNodes, clusterData, posInfo, sizeData, countData);
}

#endif  // THC_GENERIC_FILE
