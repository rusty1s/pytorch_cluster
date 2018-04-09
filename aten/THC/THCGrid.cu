#include "THCGrid.h"

#include "common.cuh"
#include "THCNumerics.cuh"

template<typename T>
__global__ void gridKernel(int64_t *self, TensorInfo<T> posInfo, T *size,
                           int64_t *count, ptrdiff_t nNodes) {
  KERNEL_LOOP(i, nNodes) {
    T *pos = posInfo.data + i * posInfo.stride[0];
    T c;
    int64_t coef = 1, value = 0;
    for (ptrdiff_t d = 0; d < posInfo.size[1]; d += posInfo.stride[1]) {
      c = THCNumerics<T>::div(pos[d], size[d]);
      c = ScalarConvert<int64_t, T>::to(ScalarConvert<T, int64_t>::to(c));
      value += coef * c;
      coef *= count[d];
    }
    self[i] = value;
  }
}

#include "generic/THCGrid.cu"
#include "THC/THCGenerateAllTypes.h"
