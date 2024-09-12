#pragma once

#include "../extensions.h"

torch::Tensor fps_cpu(torch::Tensor src, torch::Tensor ptr, torch::Tensor ratio,
                      torch::Tensor num_points, bool random_start);
