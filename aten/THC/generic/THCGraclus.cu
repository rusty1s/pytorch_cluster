#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/THCGraclus.cu"
#else

void THCTensor_(graclus)(THCState *state, THCudaLongTensor *self, THCudaLongTensor *row,
                         THCudaLongTensor *col, THCTensor *weight) {
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 4, self, row, col, weight));

  THC_TENSOR_GRACLUS(state, self, row,
    while(!THCudaLongTensor_color(state, self)) {
      THCTensor_(propose)(state, self, prop, row, col, weight, degree, cumDegree);
      THCTensor_(response)(state, self, prop, row, col, weight, degree, cumDegree);
    }
  )
}

#endif  // THC_GENERIC_FILE
