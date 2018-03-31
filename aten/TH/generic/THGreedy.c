#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/THGreedy.c"
#else

void THGreedy_(THLongTensor *cluster, THLongTensor *row, THLongTensor *col, THTensor *weight) {
  real *weightData = THTensor_getData(weight);
  real maxWeight = 0, tmpWeight;
  TH_GREEDY_CLUSTER(cluster, row, col,
    tmpWeight = weightData[idx];
    if (tmpWeight > maxWeight) {
      pairValue = colValue;
      maxWeight = tmpWeight;
    }
  )
}

#endif  // TH_GENERIC_FILE
