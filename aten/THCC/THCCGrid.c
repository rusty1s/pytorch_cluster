#include <THC/THC.h>

#include "THC.h"

#define THCCTensor_(NAME) TH_CONCAT_4(THCC,Real,Tensor_,NAME)

extern THCState *state;

#include "generic/THCCGrid.c"
#include "THCGenerateAllTypes.h"
