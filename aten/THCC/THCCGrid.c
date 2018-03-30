#include <THC/THC.h>

#include "THCGrid.h"

#define THCCGrid_ TH_CONCAT_3(THCC,Real,Grid)
#define THCGrid_ TH_CONCAT_3(TH,CReal,Grid)

extern THCState *state;

#include "generic/THCCGrid.c"
#include "THCGenerateAllTypes.h"
