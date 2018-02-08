#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/cpu.c"
#else

int64_t cluster_(grid)(THLongTensor *output, THTensor *position, THTensor *size, THTensor *maxPosition) {
  real *size_data = size->storage->data + size->storageOffset;
  real *maxPosition_data = maxPosition->storage->data + maxPosition->storageOffset;

  int64_t Dims = THTensor_(nDimension)(position);
  int64_t D = THTensor_(size)(position, Dims - 1);

  TH_TENSOR_DIM_APPLY2(int64_t, output, real, position, Dims - 1,
    int weight = 1; int64_t cluster = 0;
    for (int d = D - 1; d >= 0; d--) {
      cluster += weight * (int64_t) (*(position_data + d * position_stride) / *(size_data + d));
      weight *= (int64_t) (maxPosition_data[d] / size_data[d]) + 1;
    }
    output_data[0] = cluster;
  )

  int64_t C = 1;
  for (int d = 1; d < D; d++) {
    C *= (int64_t) (maxPosition_data[d] / size_data[d]) + 1;
  }
  return C;
}

#endif
