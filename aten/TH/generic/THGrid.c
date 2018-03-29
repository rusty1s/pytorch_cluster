#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/THGrid.c"
#else

void THGrid_(cluster)(THLongTensor *cluster, THTensor *pos, THTensor *size, THLongTensor *count) {
  real *sizeData = size->storage->data + size->storageOffset;
  int64_t *countData = count->storage->data + count->storageOffset;
  int64_t dims = THLongTensor_nElement(count);
  THLongTensor_unsqueeze1d(cluster, NULL, 1);
  ptrdiff_t d; int64_t coef, value;
  TH_TENSOR_DIM_APPLY2(int64_t, cluster, real, pos, 1,
    coef = 1; value = 0;
    for (d = 0; d < dims; d++) {
      value += coef * (int64_t) (*(pos_data + d * pos_stride) / sizeData[d]);
      coef *= countData[d];
    }
    cluster_data[0] = value;
  )
  THLongTensor_squeeze1d(cluster, NULL, 1);
}

#endif  // TH_GENERIC_FILE
