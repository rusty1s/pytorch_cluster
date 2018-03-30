#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/THGrid.c"
#else

void THGrid_(cluster)(THLongTensor *cluster, THTensor *pos, THTensor *size, THLongTensor *count) {
  int64_t *clusterData = DATA(cluster);
  real *posData = DATA(pos);
  real *sizeData = DATA(size);
  int64_t *countData = DATA(count);

  ptrdiff_t n, d; int64_t coef, value;
  for (n = 0; n < THTensor_(size)(pos, 0); n++) {
    coef = 1; value = 0;
    for (d = 0; d < THTensor_(size)(pos, 1); d++) {
      value += coef * (int64_t) (*(posData + d * pos->stride[1]) / sizeData[d]);
      coef *= countData[d];
    }
    posData += pos->stride[0];
    clusterData[n] = value;
  }
}

#endif  // TH_GENERIC_FILE
