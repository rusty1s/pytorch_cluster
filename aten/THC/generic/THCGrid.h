#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/THCGrid.h"
#else

void THCTensor_(grid)(THCState *state, THCudaLongTensor *self, THCTensor *pos, THCTensor *size,
                      THCudaLongTensor *count);

#endif  // THC_GENERIC_FILE
