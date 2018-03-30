#include <TH/TH.h>

#define THGreedy_(NAME) TH_CONCAT_4(TH,Real,Greedy_,NAME)
#define DATA(TENSOR) TENSOR->storage->data + TENSOR->storageOffset

#define TH_GREEDY_CLUSTER(cluster, row, col, deg, SELECT) { \
  THLongTensor_fill(cluster, -1); \
  int64_t *clusterData = DATA(cluster); \
  int64_t *rowData = DATA(row); \
  int64_t *colData = DATA(col); \
  int64_t *degData = DATA(deg); \
  ptrdiff_t rowIdx = 0, neighborIdx; \
  int64_t rowValue, colValue, clusterValue, tmp; \
  while(rowIdx < THLongTensor_nElement(row)) { \
    rowValue = rowData[rowIdx]; \
    printf("rowValue = %lli, ", rowValue); \
    if (clusterData[rowValue] < 0) { \
      colValue = rowValue; \
      SELECT \
      clusterValue = rowValue < colValue ? rowValue : colValue; \
      printf("%lli", clusterValue); \
      clusterData[rowValue] = clusterValue; \
      clusterData[colValue] = clusterValue; \
    } \
    rowIdx += degData[rowValue]; \
  } \
}

void THGreedy_cluster(THLongTensor *cluster, THLongTensor *row, THLongTensor *col,
                      THLongTensor *deg) {
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
