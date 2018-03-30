#include "THCGrid.h"

template<typename real, int dims>
__global__ void gridKernel(int64_t *cluster, TensorInfo<real> posInfo, real *size,
                           int64_t *count, const int nNodes) {
  KERNEL_LOOP(i, nNodes) {
    real *pos = posInfo->data + i * posInfo->stride[0];
    int64_t coef = 1, value = 0;
    for (ptrdiff_t d = 0; d < dims; d++) {
      value += coef * (int64_t) (pos[d * posInfo->stride[1]] / size[d]);
      coef *= count[d];
    }
    cluster[i] = value;
  }
}

#include "generic/THCGrid.cu"
#include "THC/THCGenerateAllTypes.h"
