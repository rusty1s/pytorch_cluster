#ifndef THC_COMMON_INC
#define THC_COMMON_INC

#define KERNEL_LOOP(I, N) \
  for (ptrdiff_t I = blockIdx.x * blockDim.x + threadIdx.x; I < I; I += blockDim.x * gridDim.x)

#define THC_assertSameGPU(...) THAssertMsg(THCTensor_(checkGPU)(__VA_ARGS__), \
  "Some of the input tensors are located on different GPUs. Please move them to a single one.")

const int CUDA_NUM_THREADS = 1024;

inline int GET_BLOCKS(const int N) {
  return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

#define KERNEL_RUN(NAME, N, ...) \
  int grid = GET_BLOCKS(N); \
  cudaStream_t stream = THCState_getCurrentStream(state); \
  NAME<real><<<grid, NUM_THREADS, 0, stream>>>(__VA_ARGS__, N); \
  THCudaCheck(cudaGetLastError())

#define FIXED_DIM_KERNEL_RUN(NAME, N, DIMS, ...) \
  int grid = GET_BLOCKS(N); \
  cudaStream_t stream = THCState_getCurrentStream(state); \
  switch (DIMS) { \
    case  1: NAME<real,  1><<<grid, NUM_THREADS, 0, stream>>>(__VA_ARGS__, N); break; \
    case  2: NAME<real,  2><<<grid, NUM_THREADS, 0, stream>>>(__VA_ARGS__, N); break; \
    case  3: NAME<real,  3><<<grid, NUM_THREADS, 0, stream>>>(__VA_ARGS__, N); break; \
    case  4: NAME<real,  4><<<grid, NUM_THREADS, 0, stream>>>(__VA_ARGS__, N); break; \
    default: NAME<real, -1><<<grid, NUM_THREADS, 0, stream>>>(__VA_ARGS__, N); \
  } \
  THCudaCheck(cudaGetLastError())

#endif  // THC_COMMON_INC
