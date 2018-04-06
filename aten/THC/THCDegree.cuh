#ifndef THC_DEGREE_INC
#define THC_DEGREE_INC

#include "common.cuh"
#include "THCNumerics.cuh"

template<typename T>
__global__ void cumDegreeKernel(T *self, int64_t *index, ptrdiff_t nEdges) {
  KERNEL_LOOP(i, nEdges) {
    int64_t r = index[i];
    if (i + 1 == nEdges) {self[r] = ScalarConvert<int, T>::to(nEdges);; continue; }
    if (r != index[i+1]) { self[r] = ScalarConvert<int, T>::to(i + 1); }
  }
}

#include "generic/THCDegree.cuh"
#include "THC/THCGenerateAllTypes.h"

#endif  // THC_DEGREE_INC
