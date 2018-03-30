#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/THCGrid.h"
#else

void THCGrid_(THCState *state, THCudaLongTensor *cluster, THCTensor *pos, THCTensor *size,
              THCudaLongTensor *count);

#endif  // THC_GENERIC_FILE
