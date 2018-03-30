#include <TH/TH.h>

#include "common.h"

#define THGreedy_ TH_CONCAT_3(TH,Real,Greedy)

#define TH_GREEDY_CLUSTER(cluster, row, col, deg, SELECT) { \
  THLongTensor_fill(cluster, -1); \
  int64_t *clusterData = THTensor_getData(cluster); \
  int64_t *rowData = THTensor_getData(row); \
  int64_t *colData = THTensor_getData(col); \
  int64_t *degData = THTensor_getData(deg); \
  ptrdiff_t rowIdx = 0, neighborIdx; \
  int64_t rowValue, colValue, clusterValue, tmp; \
  while(rowIdx < THLongTensor_nElement(row)) { \
    rowValue = rowData[rowIdx]; \
    if (clusterData[rowValue] < 0) { \
      colValue = rowValue; \
      SELECT \
      clusterValue = rowValue < colValue ? rowValue : colValue; \
      clusterData[rowValue] = clusterValue; \
      clusterData[colValue] = clusterValue; \
    } \
    rowIdx += degData[rowValue]; \
  } \
}

void THGreedy(THLongTensor *cluster, THLongTensor *row, THLongTensor *col, THLongTensor *deg) {
  TH_GREEDY_CLUSTER(cluster, row, col, deg,
    for (neighborIdx = rowIdx; neighborIdx < rowIdx + degData[rowValue]; neighborIdx++) {
      tmp = colData[neighborIdx];
      if (clusterData[tmp] < 0) {
        colValue = tmp;
        break;
      }
    }
  )
}

#include "generic/THGreedy.c"
#include "THGenerateAllTypes.h"
