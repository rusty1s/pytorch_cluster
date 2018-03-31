#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/THCDegree.cu"

#else
void THCTensor_(degree)(THCState *state, THCTensor *self, THCudaLongTensor *index) {
  int nEdges = THCudaLongTensor_nElement(state, index);
  THCTensor *one = THCTensor_(newWithSize1d)(state, nEdges);
  THCTensor_(fill)(state, one, 1);

  THCTensor_(fill)(state, self, 0);
  THCTensor_(scatterAdd)(state, self, 0, index, one);

  THCTensor_(free)(state, one);
}

#endif  // THC_GENERIC_FILE
