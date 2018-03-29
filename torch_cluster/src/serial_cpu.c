#include <TH/TH.h>

#define cluster_(NAME) TH_CONCAT_4(cluster_, NAME, _, Real)

#define SERIAL(output, row, col, degree, SELECT) { \
  int64_t *output_data = output->storage->data + output->storageOffset; \
  int64_t *row_data = row->storage->data + row->storageOffset; \
  int64_t *col_data = col->storage->data + col->storageOffset; \
  int64_t *degree_data = degree->storage->data + degree->storageOffset; \
  \
  int64_t e = 0, row_value, col_value, v; \
  while(e < THLongTensor_nElement(row)) { \
    row_value = row_data[e]; \
    if (output_data[row_value] < 0) { \
      col_value = -1; \
      SELECT \
      if (col_value < 0) { \
        output_data[row_value] = row_value; \
      } \
      else { \
        v = row_value < col_value ? row_value : col_value; \
        output_data[row_value] = v; \
        output_data[col_value] = v; \
      } \
    } \
    e += degree_data[row_value]; \
  } \
}

void cluster_serial(THLongTensor *output, THLongTensor *row, THLongTensor *col, THLongTensor *degree) {
  int64_t d, c;
  SERIAL(output, row, col, degree,
    for (d = 0; d < degree_data[row_value]; d++) {  // Iterate over neighbors.
      c = col_data[e + d];
      if (output_data[c] < 0) {  // Neighbor is unmatched.
        col_value = c;
        break;
      }
    }
  )
}

#include "generic/serial_cpu.c"
#include "THGenerateAllTypes.h"
