#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/THCCGrid.c"
#else

void THCCTensor_(grid)(THCudaLongTensor *self, THCTensor *pos, THCTensor *size,
                       THCudaLongTensor *count) {
  THCTensor_(grid)(state, self, pos, size, count);
}

#endif  // THC_GENERIC_FILE
