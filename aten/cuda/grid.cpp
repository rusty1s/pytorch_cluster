#include <torch/torch.h>

at::Tensor grid_cuda(at::Tensor pos, at::Tensor size, at::Tensor start,
                     at::Tensor end);

at::Tensor grid(at::Tensor pos, at::Tensor size, at::Tensor start,
                at::Tensor end) {
  CHECK_CUDA(pos);
  CHECK_CUDA(size);
  CHECK_CUDA(start);
  CHECK_CUDA(end);

  return grid_cuda(pos, size, start, end);
}
