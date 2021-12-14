#pragma once

#include <torch/extension.h>

#define CHECK_CUDA(x)                                                          \
  AT_ASSERTM(x.device().is_cuda(), #x " must be CUDA tensor")
#define CHECK_INPUT(x) AT_ASSERTM(x, "Input mismatch")
#define CHECK_CONTIGUOUS(x)                                                    \
  AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")

__device__ int64_t get_example_idx(int64_t idx, const int64_t *ptr,
                                   const int64_t num_examples) {
  for (int64_t i = 0; i < num_examples; i++) {
    if (ptr[i + 1] > idx)
      return i;
  }
  return num_examples - 1;
}
