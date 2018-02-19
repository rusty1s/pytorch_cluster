#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/kernel.cu"
#else

void cluster_(grid)(THCState *state, int C, THCudaLongTensor *output, THCTensor *position, THCTensor *size, THCudaLongTensor *count) {
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 2, position, size));
  THCAssertSameGPU(THCudaLongTensor_checkGPU(state, 2, output, count));
  THArgCheck(THCudaLongTensor_nDimension(state, output) <= MAX_DIMS, 1, "Tensor too large or too many dimensions");

  int64_t *outputData = THCudaLongTensor_data(state, output);
  TensorInfo<real> positionInfo = thc_(getTensorInfo)(state, position);
  real *sizeData = THCTensor_(data)(state, size);
  int64_t *countData = THCudaLongTensor_data(state, count);

  const int N = THCudaLongTensor_nElement(state, output);
  int grid = GET_BLOCKS(N);
  cudaStream_t stream = THCState_getCurrentStream(state);

  switch (positionInfo.dims) {
    case  1: gridKernel<real,  1><<<grid, NUM_THREADS, 0, stream>>>(outputData, positionInfo, sizeData, countData, C, N); break;
    case  2: gridKernel<real,  2><<<grid, NUM_THREADS, 0, stream>>>(outputData, positionInfo, sizeData, countData, C, N); break;
    case  3: gridKernel<real,  3><<<grid, NUM_THREADS, 0, stream>>>(outputData, positionInfo, sizeData, countData, C, N); break;
    case  4: gridKernel<real,  4><<<grid, NUM_THREADS, 0, stream>>>(outputData, positionInfo, sizeData, countData, C, N); break;
    default: gridKernel<real, -1><<<grid, NUM_THREADS, 0, stream>>>(outputData, positionInfo, sizeData, countData, C, N); break;
  }

  THCudaCheck(cudaGetLastError());
}

#endif
