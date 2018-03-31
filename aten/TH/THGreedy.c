#include <TH/TH.h>

#include "common.h"

#define THGreedy_ TH_CONCAT_3(TH,Real,Greedy)

#define TH_GREEDY_CLUSTER(cluster, row, col, SELECT) { \
  THLongTensor_fill(cluster, -1); \
  int64_t *clusterData = THTensor_getData(cluster); \
  int64_t *rowData = THTensor_getData(row); \
  int64_t *colData = THTensor_getData(col); \
  ptrdiff_t idx = 0, nEdges = THLongTensor_nElement(row); \
  int64_t rowValue, pairValue, colValue, clusterValue; \
  while(idx < nEdges) { \
    rowValue = rowData[idx]; \
    pairValue = rowValue; \
    if (clusterData[rowValue] < 0) { \
      while(idx < nEdges && rowData[idx] == rowValue) { \
        colValue = colData[idx]; \
        if (clusterData[colValue] < 0) { \
          SELECT \
        } \
        idx++; \
      } \
    } \
    clusterValue = rowValue < pairValue ? rowValue : pairValue; \
    clusterData[rowValue] = clusterValue; \
    clusterData[pairValue] = clusterValue; \
    while(idx < nEdges && rowData[idx] == rowValue) idx++; \
  } \
}

void THGreedy(THLongTensor *cluster, THLongTensor *row, THLongTensor *col) {
  TH_GREEDY_CLUSTER(cluster, row, col,
    pairValue = colValue;
    break;
  )
}

#include "generic/THGreedy.c"
#include "THGenerateAllTypes.h"
