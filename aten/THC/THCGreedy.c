#include <THC/THC.h>

#define THCGreedy_(NAME) TH_CONCAT_4(TH,CReal,Greedy_,NAME)

extern THCState *state;

void THCGreedy_cluster(THCudaLongTensor *cluster, THCudaLongTensor *row, THCudaLongTensor *col,
                       THCudaLongTensor *deg) {
}

#include "generic/THCGreedy.c"
#include "THCGenerateAllTypes.h"

