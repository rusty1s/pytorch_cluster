#include <TH/TH.h>

#define cluster_(NAME) TH_CONCAT_4(cluster_, NAME, _, Real)

void cluster_serial(THLongTensor *output, THLongTensor *row, THLongTensor *col) {
  int64_t *output_data = output->storage->data + output->storageOffset;
  int64_t *row_data = row->storage->data + row->storageOffset;
  int64_t *col_data = col->storage->data + col->storageOffset;
  int64_t n, N = THLongTensor_nElement(output), r, c, value;
  for (n = 0; n < N; n++) {
    r = row_data[n]; c = col_data[c];
    if (output_data[r] == -1 && output_data[c] == -1) {
      value = r < c ? r : c;
      output_data[r] = value;
      output_data[c] = value;
    }
  }
}

#include "generic/cpu.c"
#include "THGenerateAllTypes.h"
