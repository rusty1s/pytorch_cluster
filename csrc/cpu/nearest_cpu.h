#pragma once

#include "../extensions.h"

torch::Tensor nearest_cpu(torch::Tensor x, torch::Tensor y,
                          torch::Tensor batch_x, torch::Tensor batch_y);
