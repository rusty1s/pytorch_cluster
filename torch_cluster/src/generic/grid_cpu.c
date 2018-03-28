#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/grid_cpu.c"
#else

void cluster_(grid)(int C, THLongTensor *output, THTensor *position, THTensor *size, THLongTensor *count) {
  real *size_data = size->storage->data + size->storageOffset;
  int64_t *count_data = count->storage->data + count->storageOffset;

  int64_t D = THLongTensor_nElement(count), d, c, tmp;
  TH_TENSOR_DIM_APPLY2(int64_t, output, real, position, THTensor_(nDimension)(position) - 1,
    tmp = C; c = 0;
    for (d = 0; d < D; d++) {
      tmp = tmp / *(count_data + d);
      c += tmp * (int64_t) (*(position_data + d * position_stride) / *(size_data + d));
    }
    output_data[0] = c;
  )
}

#endif
