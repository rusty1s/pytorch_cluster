#ifndef THC_GREEDY_INC
#define THC_GREEDY_INC

#include <THC/THC.h>

#define THCGreedy_(NAME) TH_CONCAT_4(TH,CReal,Greedy_,NAME)

void THCGreedy_cluster(THCState *state,
                       THCudaLongTensor *cluster,
                       THCudaLongTensor *row,
                       THCudaLongTensor *col,
                       THCudaLongTensor *deg);

#include "generic/THCGreedy.h"
#include "THC/THCGenerateAllTypes.h"

#endif  // THC_GREEDY_INC
