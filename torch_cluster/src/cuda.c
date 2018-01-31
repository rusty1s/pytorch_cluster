#include <THC/THC.h>

#include "kernel.h"

#define cluster_(NAME) TH_CONCAT_4(cluster_, NAME, _cuda_, Real)
#define cluster_kernel_(NAME) TH_CONCAT_4(cluster_, NAME, _kernel_, Real)

extern THCState *state;

#include "generic/cuda.c"
#include "THCGenerateFloatType.h"
#include "generic/cuda.c"
#include "THCGenerateDoubleType.h"
#include "generic/cuda.c"
#include "THCGenerateByteType.h"
#include "generic/cuda.c"
#include "THCGenerateCharType.h"
#include "generic/cuda.c"
#include "THCGenerateShortType.h"
#include "generic/cuda.c"
#include "THCGenerateIntType.h"
#include "generic/cuda.c"
#include "THCGenerateLongType.h"
