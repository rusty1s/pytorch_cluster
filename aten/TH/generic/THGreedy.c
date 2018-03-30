#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/THGreedy.c"
#else

void THGreedy_(THLongTensor *cluster, THLongTensor *row, THLongTensor *col, THLongTensor *deg,
               THTensor *weight) {
  real *weightData = DATA(weight);
  real maxWeight = 0, tmpWeight;
  TH_GREEDY_CLUSTER(cluster, row, col, deg,
    for (neighborIdx = rowIdx; neighborIdx < rowIdx + degData[rowValue]; neighborIdx++) {
      tmp = colData[neighborIdx];
      tmpWeight = weightData[neighborIdx];
      if (clusterData[tmp] < 0 && tmpWeight > maxWeight) {
        colValue = tmp;
        maxWeight = tmpWeight;
      }
    }
  )
}

#endif  // TH_GENERIC_FILE
