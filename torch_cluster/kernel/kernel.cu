#include <THC.h>

#include "kernel.h"

#include "common.cuh"
#include "THCIndex.cuh"

#define cluster_(NAME) TH_CONCAT_4(cluster_, NAME, _kernel_, Real)
#define thc_(NAME) TH_CONCAT_4(thc_, NAME, _, Real)

#include "generic/common.cu"
#include "THCGenerateAllTypes.h"

template<typename Real, int Dims>
__global__ void gridKernel(int64_t *output, TensorInfo<Real> position, Real *size, int64_t *count, const int C, const int N) {
  KERNEL_LOOP(i, N) {
    int positionOffset = 0;
    IndexToOffset<Real, Dims>::compute(i, position, &positionOffset);

    int tmp = C; int64_t c = 0;
    for (int d = 0; d < position.size[position.dims - 1]; d++) {
      tmp = tmp / count[d];
      c += tmp * (int64_t) (position.data[positionOffset + d] / size[d]);
    }
    output[i] = c;
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
