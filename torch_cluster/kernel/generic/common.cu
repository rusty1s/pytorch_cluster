#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/common.cu"
#else

TensorInfo<real> thc_(getTensorInfo)(THCState *state, THCTensor *tensor) {
  real *data = THCTensor_(data)(state, tensor);
  int dims = THCTensor_(nDimension)(state, tensor);
  int size[MAX_DIMS]; int stride[MAX_DIMS];
  for (int i = 0; i < dims; i++) {
    size[i] = THCTensor_(size)(state, tensor, i);
    stride[i] = THCTensor_(stride)(state, tensor, i);
  }
  return TensorInfo<real>(data, dims, size, stride);
}

#endif
