#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/THGraclus.c"
#else

void THTensor_(graclus)(THLongTensor *self, THLongTensor *row, THLongTensor *col, THTensor *weight) {
  real *weightData = THTensor_(data)(weight);
  real maxWeight, tmp;
  TH_TENSOR_GRACLUS(self, row, col, maxWeight = 0;,
    tmp = weightData[e];
    if (selfData[colValue] < 0 && tmp > maxWeight) { matchedValue = colValue; maxWeight = tmp; }
  )
}

#endif  // TH_GENERIC_FILE
