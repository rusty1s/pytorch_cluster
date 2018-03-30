#ifndef THC_COMMON_INC
#define THC_COMMON_INC

#define THCTensor_(NAME) TH_CONCAT_4(TH,CReal,Tensor_,NAME)

#define KERNEL_LOOP(I, N) \
  for (ptrdiff_t I = blockIdx.x * blockDim.x + threadIdx.x; I < N; I += blockDim.x * gridDim.x)

#define THC_assertSameGPU(...) THAssertMsg(THCTensor_(checkGPU)(__VA_ARGS__), \
  "Some of the input tensors are located on different GPUs. Please move them to a single one.")

const int MAX_DIMS = 25;
const int NUM_THREADS = 1024;

inline int GET_BLOCKS(const int N) {
  return (N + NUM_THREADS - 1) / NUM_THREADS;
}

#define KERNEL_RUN(NAME, N, ...) \
  int grid = GET_BLOCKS(N); \
  cudaStream_t stream = THCState_getCurrentStream(state); \
  NAME<real><<<grid, NUM_THREADS, 0, stream>>>(__VA_ARGS__, N); \
  THCudaCheck(cudaGetLastError())

template<typename T>
struct TensorInfo {
  T *data;
  int dims;
  int size[MAX_DIMS];
  int stride[MAX_DIMS];
};

#include "generic/common.h"
#include "THC/THCGenerateAllTypes.h"

#endif  // THC_COMMON_INC
