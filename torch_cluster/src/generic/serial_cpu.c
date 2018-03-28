#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/serial_cpu.c"
#else

void cluster_(serial)(THLongTensor *output, THLongTensor *row, THLongTensor *col, THTensor *weight, THLongTensor *degree) {
  real *weight_data = weight->storage->data + weight->storageOffset;
  real max_weight, w;
  int64_t d, c;
  SERIAL(output, row, col, degree,
    max_weight = 0;
    for (d = 0; d < degree_data[row_value]; d++) {
      c = col_data[e + d];
      w = weight_data[e + d];
      if (output_data[c] < 0 && w >= max_weight) {
        col_value = c;
        max_weight = w;
      }
    }
  )
}

#endif
