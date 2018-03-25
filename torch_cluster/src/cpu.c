#include <TH/TH.h>

#define cluster_(NAME) TH_CONCAT_4(cluster_, NAME, _, Real)

void cluster_random(THLongTensor *output, THLongTensor *row, THLongTensor *col, THLongTensor *degree) {
  /* int64_t *output_data = output->storage->data + output->storageOffset; */
  /* int64_t *row_data = row->storage->data + row->storageOffset; */
  /* int64_t *col_data = col->storage->data + col->storageOffset; */
  /* int64_t e, E = THLongTensor_nElement(row), r, c, value; */
  /* for (e = 0; e < E; e++) { */
  /*   r = row_data[e]; c = col_data[e]; */
  /*   if (output_data[r] == -1 && output_data[c] == -1) { */
  /*     value = r < c ? r : c; */
  /*     output_data[r] = value; */
  /*     output_data[c] = value; */
  /*   } */
  /* } */
}

#include "generic/cpu.c"
#include "THGenerateAllTypes.h"
