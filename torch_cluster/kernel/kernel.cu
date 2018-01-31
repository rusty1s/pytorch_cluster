#include <THC.h>

#include "kernel.h"

#define cluster_(NAME) TH_CONCAT_4(cluster, NAME, _kernel_, Real)

#include "generic/kernel.cu"
#include "THCGenerateAllTypes.h"
