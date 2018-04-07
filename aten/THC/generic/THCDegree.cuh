#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/THCDegree.cuh"
#else

void THCTensor_(degree)(THCState *state, THCTensor *self, THCudaLongTensor *index) {
  int nEdges = THCudaLongTensor_nElement(state, index);

  THCTensor *one = THCTensor_(newWithSize1d)(state, nEdges);
  THCTensor_(fill)(state, one, ScalarConvert<int, real>::to(1));

  THCTensor_(fill)(state, self, ScalarConvert<int, real>::to(0));
  THCTensor_(scatterAdd)(state, self, 0, index, one);

  THCTensor_(free)(state, one);
}

void THCTensor_(cumDegree)(THCState *state, THCTensor *self, THCudaLongTensor *index) {
  ptrdiff_t nEdges = THCudaLongTensor_nElement(state, index);

  real *selfData = THCTensor_(data)(state, self);
  int64_t *indexData = THCudaLongTensor_data(state, index);

  KERNEL_REAL_RUN(cumDegreeKernel, nEdges, selfData, indexData);
}

#endif  // THC_GENERIC_FILE
