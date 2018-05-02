#ifndef PERM_INC
#define PERM_INC

#include <torch/torch.h>

inline std::tuple<at::Tensor, at::Tensor>
randperm(at::Tensor row, at::Tensor col, int num_nodes);

#endif // PERM_INC
