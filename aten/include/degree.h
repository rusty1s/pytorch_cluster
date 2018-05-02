#ifndef DEGREE_INC
#define DEGREE_INC

#include <torch/torch.h>

inline at::Tensor degree(at::Tensor index, int num_nodes,
                         at::ScalarType scalar_type);

#endif // DEGREE_INC
