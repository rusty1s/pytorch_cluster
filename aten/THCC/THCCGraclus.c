#include <THC/THC.h>

#include "THC.h"

#define THCCTensor_(NAME) TH_CONCAT_4(THCC,Real,Tensor_,NAME)

extern THCState *state;

void THCCTensor_graclus(THCudaLongTensor *self, THCudaLongTensor *row, THCudaLongTensor *col) {
  THCTensor_graclus(state, self, row, col);
}

#include "generic/THCCGraclus.c"
#include "THCGenerateAllTypes.h"
