#include <TH/TH.h>

#define TH_TENSOR_GRACLUS(self, row, col, PRESELECT, SELECT) { \
  THLongTensor_fill(self, -1); \
  int64_t *selfData = THLongTensor_data(self); \
  int64_t *rowData = THLongTensor_data(row); \
  int64_t *colData = THLongTensor_data(col); \
  ptrdiff_t e = 0, nEdges = THLongTensor_nElement(row); \
  int64_t rowValue, colValue, matchedValue, value; \
  while(e < nEdges) { \
    rowValue = rowData[e]; \
    matchedValue = rowValue; \
    PRESELECT \
    if (selfData[rowValue] < 0) { \
      do { \
        colValue = colData[e]; \
        SELECT \
        e++; \
      } while(e < nEdges && rowData[e] == rowValue); \
      value = rowValue < matchedValue ? rowValue : matchedValue; \
      selfData[rowValue] = value; \
      selfData[matchedValue] = value; \
    } \
    while(e < nEdges && rowData[e] == rowValue) e++; \
  } \
}

void THTensor_graclus(THLongTensor *self, THLongTensor *row, THLongTensor *col) {
  TH_TENSOR_GRACLUS(self, row, col, {},
    if (selfData[colValue] < 0) { matchedValue = colValue; break; }
  )
}

#include "generic/THGraclus.c"
#include "THGenerateAllTypes.h"
