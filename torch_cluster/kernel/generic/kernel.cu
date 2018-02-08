#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/kernel.cu"
#else

int64_t cluster_(grid)(THCState *state, THCudaLongTensor *output, THCTensor *position, THCTensor *size, THCTensor *maxPosition) {
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 3, position, size, maxPosition));
  THCAssertSameGPU(THCudaLongTensor_checkGPU(state, 1, output));
  THArgCheck(THCTensor_(nDimension)(state, position) <= MAX_DIMS, 1, "Tensor too large or too many dimensions");

  int64_t *outputData = THCudaLongTensor_data(state, output);
  TensorInfo<real> positionInfo = thc_(getTensorInfo)(state, position);
  real *sizeData = THCTensor_(data)(state, size);
  real *maxPositionData = THCTensor_(data)(state, maxPosition);

  const int N = THCudaLongTensor_nElement(state, output);
  int grid = GET_BLOCKS(N);
  cudaStream_t stream = THCState_getCurrentStream(state);

  switch (positionInfo.dims) {
    case  1: gridKernel<real,  1><<<grid, NUM_THREADS, 0, stream>>>(outputData, positionInfo, sizeData, maxPositionData, N); break;
    case  2: gridKernel<real,  2><<<grid, NUM_THREADS, 0, stream>>>(outputData, positionInfo, sizeData, maxPositionData, N); break;
    case  3: gridKernel<real,  3><<<grid, NUM_THREADS, 0, stream>>>(outputData, positionInfo, sizeData, maxPositionData, N); break;
    default: gridKernel<real, -1><<<grid, NUM_THREADS, 0, stream>>>(outputData, positionInfo, sizeData, maxPositionData, N); break;
  }

  THCudaCheck(cudaGetLastError());

  real C = 1;
  for (ptrdiff_t d = 1; d < THCTensor_(nElement)(state, size); d++) {
    C = maxPositionData[d] / sizeData[d];
    /* printf("%f", maxPositionData[d]); */
    /* printf("%i", (int)*(maxPositionData)); */
    /* C *= (int64_t) (*(maxPositionData + d) / *(sizeData + d)) + 1; */
  }
  return C;
}

#endif
