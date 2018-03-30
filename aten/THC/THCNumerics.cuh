#ifndef THC_NUMERICS_INC
#define THC_NUMERICS_INC

#include "THC/THCHalf.h"

template<typename T>
struct THCNumerics {
  static inline __host__ __device__ T div(T a, T b) { return a / b; }
  static inline __host__ __device__ int floor(T a) { return a; }
};

#ifdef CUDA_HALF_TENSOR
#ifdef __CUDA_ARCH__
#define h2f(A) __half2float(A)
#define f2h(A) __float2half(A)
#else  // CUDA_ARCH__
#define h2f(A) THC_half2float(A)
#define f2h(A) THC_float2half(A)
#endif
template<>
struct THCNumerics<half> {
  static inline __host__ __device__ half div(half a, half b) { return f2h(h2f(a) / h2f(b)); }
  static inline __host__ __device__ int floor(half a) { return (int) h2f(a); }
};
#endif  // CUDA_HALF_TENSOR

#endif  // THC_NUMERICS_INC
