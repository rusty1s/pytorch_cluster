#ifndef LOOP_INC
#define LOOP_INC

#include <torch/torch.h>

inline std::tuple<at::Tensor, at::Tensor> remove_self_loops(at::Tensor row,
                                                            at::Tensor col) {
  auto mask = row != col;
  return {row.masked_select(mask), col.masked_select(mask)};
}

#endif // LOOP_INC
