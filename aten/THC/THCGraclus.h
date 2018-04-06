#ifndef THC_GRACLUS_INC
#define THC_GRACLUS_INC

#include <THC/THC.h>

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

void THCTensor_graclus(THCState *state, THCudaLongTensor *self, THCudaLongTensor *row,
                       THCudaLongTensor *col);

#include "generic/THCGraclus.h"
#include "THC/THCGenerateAllTypes.h"

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // THC_GRACLUS_INC
