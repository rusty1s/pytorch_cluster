#include <THC.h>

#include "kernel.h"

#include "common.cuh"
#include "THCIndex.cuh"

#define cluster_(NAME) TH_CONCAT_4(cluster_, NAME, _kernel_, Real)
#define thc_(NAME) TH_CONCAT_4(thc_, NAME, _, Real)

#include "generic/common.cu"
#include "THCGenerateAllTypes.h"

template<typename Real, int Dims>
__global__ void gridKernel(int64_t *output, TensorInfo<Real> position, Real *size, Real *maxPosition, const int N) {
  KERNEL_LOOP(i, N) {
    int positionOffset = 0;
    IndexToOffset<Real, Dims>::compute(i, position, &positionOffset);

    int D = position.size[position.dims - 1];
    int weight = 1; int64_t cluster = 0;
    for (int d = D - 1; d >= 0; d--) {
      cluster += weight * (int64_t) (position.data[positionOffset + d] / size[d]);
      weight *= (int64_t) (maxPosition[d] / size[d]) + 1;
    }
    output[i] = cluster;
  }
}

#include "generic/kernel.cu"
#include "THCGenerateFloatType.h"
#include "generic/kernel.cu"
#include "THCGenerateDoubleType.h"
#include "generic/kernel.cu"
#include "THCGenerateByteType.h"
#include "generic/kernel.cu"
#include "THCGenerateCharType.h"
#include "generic/kernel.cu"
#include "THCGenerateShortType.h"
#include "generic/kernel.cu"
#include "THCGenerateIntType.h"
#include "generic/kernel.cu"
#include "THCGenerateLongType.h"
