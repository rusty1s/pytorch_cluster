#ifndef THC_GREEDY_INC
#define THC_GREEDY_INC

#include <THC/THC.h>

#define THCGreedy_ TH_CONCAT_3(TH,CReal,Greedy)

void THCGreedy(THCState *state, THCudaLongTensor *cluster, THCudaLongTensor *row,
               THCudaLongTensor *col, THCudaLongTensor *deg);

#include "generic/THCGreedy.h"
#include "THC/THCGenerateAllTypes.h"

#endif  // THC_GREEDY_INC
