#include "THCGrid.h"

#include "common.cuh"
#include "THCNumerics.cuh"

template<typename T>
__global__ void gridKernel(int64_t *self, TensorInfo<T> posInfo, T *size,
                           int64_t *count, ptrdiff_t nNodes) {
  KERNEL_LOOP(i, nNodes) {
    T *pos = posInfo.data + i * posInfo.stride[0];
    int64_t coef = 1, value = 0;
    for (ptrdiff_t d = 0; d < posInfo.size[1]; d += posInfo.stride[1]) {
      value += coef * ScalarConvert<T, int64_t>::to(THCNumerics<T>::div(pos[d], size[d]));
      coef *= count[d];
    }
    self[i] = value;
  }
}

#include "generic/THCGrid.cu"
#include "THC/THCGenerateAllTypes.h"
