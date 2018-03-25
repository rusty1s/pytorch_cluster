#include <TH/TH.h>

#define cluster_(NAME) TH_CONCAT_4(cluster_, NAME, _, Real)

void cluster_random(THLongTensor *output, THLongTensor *row, THLongTensor *col, THLongTensor *degree) {
  int64_t *output_data = output->storage->data + output->storageOffset;
  int64_t *row_data = row->storage->data + row->storageOffset;
  int64_t *col_data = col->storage->data + col->storageOffset;
  int64_t *degree_data = degree->storage->data + degree->storageOffset;

  int64_t e = 0, row_value, col_value, i;

  while(e < THLongTensor_nElement(row)) {
    row_value = row_data[e];
    if (output_data[row_value] < 0) { // Node is unmatched.
      // Find next unmatched neighbor.
      col_value = -1;
      for (i = 0; i < degree_data[row_value]; i++) {
        col_value = col_data[e + i];
        if (output_data[col_value] < 0) break; // Neighbor found.
        else col_value = -1;
      }

      // Set output.
      if (col_value < 0) {
        output_data[row_value] = row_value;
      }
      else {
        i = row_value < col_value ? row_value : col_value;
        output_data[row_value] = i;
        output_data[col_value] = i;
      }
    }
    e += degree_data[row_value];
  }
}

#include "generic/cpu.c"
#include "THGenerateAllTypes.h"
