#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/THGrid.c"
#else

void THTensor_(grid)(THLongTensor *self, THTensor *pos, THTensor *size, THLongTensor *count) {
  int64_t *selfData = THLongTensor_data(self);
  real *posData = THTensor_(data)(pos);
  real *sizeData = THTensor_(data)(size);
  int64_t *countData = THLongTensor_data(count);

  ptrdiff_t n, d; int64_t coef, value;
  for (n = 0; n < THTensor_(size)(pos, 0); n++) {
    coef = 1; value = 0;
    for (d = 0; d < THTensor_(size)(pos, 1); d++) {
      value += coef * (int64_t) (posData[d * pos->stride[1]] / sizeData[d]);
      coef *= countData[d];
    }
    posData += pos->stride[0];
    selfData[n] = value;
  }
}

#endif  // TH_GENERIC_FILE
