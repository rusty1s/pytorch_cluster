#include "THCGrid.h"

#include "common.cuh"
#include "THCNumerics.cuh"

template<typename T>
__global__ void gridKernel(int64_t *cluster, TensorInfo<T> posInfo, T *size,
                           int64_t *count, ptrdiff_t nNodes) {
  KERNEL_LOOP(i, nNodes) {
    T *pos = posInfo.data + i * posInfo.stride[0];
    int64_t coef = 1, value = 0;
    for (ptrdiff_t d = 0; d < posInfo.dims * posInfo.stride[1]; d += posInfo.stride[1]) {
      value += coef * THCNumerics<T>::floor(THCNumerics<T>::div(pos[d], size[d]));
      coef *= count[d];
    }
    cluster[i] = value;
  }
}

#include "generic/THCGrid.cu"
#include "THC/THCGenerateAllTypes.h"
