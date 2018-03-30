#include <THC/THC.h>

#define THCGrid_(NAME) TH_CONCAT_4(TH,CReal,Grid_,NAME)

extern THCState *state;

#include "generic/THCGrid.c"
#include "THCGenerateAllTypes.h"
