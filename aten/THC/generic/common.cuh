#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/common.cuh"
#else

TensorInfo<real> THCTensor_(getTensorInfo)(THCState *state, THCTensor *tensor) {
  TensorInfo<real> tensorInfo = TensorInfo<real>();
  tensorInfo.data = THCTensor_(data)(state, tensor);
  tensorInfo.dims = THCTensor_(nDimension)(state, tensor);
  for (ptrdiff_t d = 0; d < tensorInfo.dims; d++) {
    tensorInfo.size[d] = THCTensor_(size)(state, tensor, d);
    tensorInfo.stride[d] = THCTensor_(stride)(state, tensor, d);
  }
  return tensorInfo;
}

#endif  // THC_GENERIC_FILE
