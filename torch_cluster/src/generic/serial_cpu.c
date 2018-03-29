#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/serial_cpu.c"
#else

void cluster_(serial)(THLongTensor *output, THLongTensor *row, THLongTensor *col, THLongTensor *degree, THTensor *weight) {
  real *weight_data = weight->storage->data + weight->storageOffset;
  real weight_value, w;
  int64_t d, c;
  SERIAL(output, row, col, degree,
    weight_value = 0;
    for (d = 0; d < degree_data[row_value]; d++) {  // Iterate over neighbors.
      c = col_data[e + d];
      w = weight_data[e + d];
      if (output_data[c] < 0 && w >= weight_value) {
        // Neighbor is unmatched and edge has a higher weight.
        col_value = c;
        weight_value = w;
      }
    }
  )
}

#endif
