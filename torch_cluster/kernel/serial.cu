#include <THC.h>

#include "serial.h"

#define cluster_(NAME) TH_CONCAT_4(cluster_, NAME, _kernel_, Real)
#define thc_(NAME) TH_CONCAT_4(thc_, NAME, _, Real)

void cluster_serial_kernel(THCState *state, THCudaLongTensor *output, THCudaLongTensor *row, THCudaLongTensor *col, THCudaLongTensor *degree) {
}

#include "generic/serial.cu"
#include "THCGenerateFloatType.h"
#include "generic/serial.cu"
#include "THCGenerateDoubleType.h"
#include "generic/serial.cu"
#include "THCGenerateByteType.h"
#include "generic/serial.cu"
#include "THCGenerateCharType.h"
#include "generic/serial.cu"
#include "THCGenerateShortType.h"
#include "generic/serial.cu"
#include "THCGenerateIntType.h"
#include "generic/serial.cu"
#include "THCGenerateLongType.h"
