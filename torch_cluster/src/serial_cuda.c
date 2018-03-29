#include <THC/THC.h>

#include "serial.h"

#define cluster_(NAME) TH_CONCAT_4(cluster_, NAME, _cuda_, Real)
#define cluster_kernel_(NAME) TH_CONCAT_4(cluster_, NAME, _kernel_, Real)

extern THCState *state;

void cluster_serial_cuda(THCudaLongTensor *output, THCudaLongTensor *row, THCudaLongTensor *col, THCudaLongTensor *degree) {
  int bla = assignColor(state, output);
  printf("RETURN TYPE IS %i \n", bla);
  /* cluster_serial_kernel(state, output, row, col, degree); */
}

#include "generic/serial_cuda.c"
#include "THCGenerateFloatType.h"
#include "generic/serial_cuda.c"
#include "THCGenerateDoubleType.h"
#include "generic/serial_cuda.c"
#include "THCGenerateByteType.h"
#include "generic/serial_cuda.c"
#include "THCGenerateCharType.h"
#include "generic/serial_cuda.c"
#include "THCGenerateShortType.h"
#include "generic/serial_cuda.c"
#include "THCGenerateIntType.h"
#include "generic/serial_cuda.c"
#include "THCGenerateLongType.h"
