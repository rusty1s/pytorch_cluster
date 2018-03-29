#include <TH/TH.h>

#define THGreedy_(NAME) TH_CONCAT_4(TH,Real,Greedy_,NAME)

#define TH_GREEDY_CLUSTER(cluster, row, col, deg, SELECT) { \
  int64_t *clusterData = cluster->storage->data + cluster->storageOffset; \
  int64_t *rowData = row->storage->data + row->storageOffset; \
  int64_t *colData = col->storage->data + col->storageOffset; \
  int64_t *degData = deg->storage->data + deg->storageOffset; \
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
