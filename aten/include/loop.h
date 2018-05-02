#ifndef LOOP_INC
#define LOOP_INC

#include <torch/torch.h>

inline std::tuple<at::Tensor, at::Tensor> remove_self_loops(at::Tensor row,
                                                            at::Tensor col);

#endif // LOOP_INC
