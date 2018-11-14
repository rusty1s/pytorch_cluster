#include <ATen/ATen.h>

#define THREADS 1024

at::Tensor nearest_cuda(at::Tensor x, at::Tensor y, at::Tensor batch_x,
                        at::Tensor batch_y) {
  return batch_x;
}
