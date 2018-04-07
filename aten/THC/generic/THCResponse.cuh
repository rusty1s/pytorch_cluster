#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/THCResponse.cuh"
#else

void THCTensor_(response)(THCState *state, THCudaLongTensor *color, THCudaLongTensor *prop,
                          THCudaLongTensor *row, THCudaLongTensor *col, THCTensor *weight,
                          THCudaLongTensor *degree, THCudaLongTensor *cumDegree) {
  KERNEL_REAL_RUN(weightedResponseKernel, THCudaLongTensor_nElement(state, color),
                  THCudaLongTensor_data(state, color), THCudaLongTensor_data(state, prop),
                  THCudaLongTensor_data(state, row), THCudaLongTensor_data(state, col),
                  THCTensor_(data)(state, weight), THCudaLongTensor_data(state, degree),
                  THCudaLongTensor_data(state, cumDegree));
}

#endif  // THC_GENERIC_FILE
