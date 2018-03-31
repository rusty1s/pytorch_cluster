void THCDegree(THCState *state, THCudaLongTensor *self, THCudaLongTensor *index) {
  int nEdges = THCudaLongTensor_nElement(state, index);
  THCudaLongTensor *one = THCudaLongTensor_newWithSize1d(state, nEdges);
  THCudaLongTensor_fill(state, one, 1);

  THCudaLongTensor_fill(state, self, 0);
  THCudaLongTensor_scatterAdd(state, self, 0, index, one);

  THCudaLongTensor_free(state, one);
}
