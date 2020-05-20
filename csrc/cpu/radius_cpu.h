#pragma once

#include <torch/extension.h>
#include "utils/neighbors.h"
#include "utils/neighbors.cpp"
#include <iostream>
#include "compat.h"

torch::Tensor radius_cpu(torch::Tensor query, torch::Tensor support,
                         torch::Tensor ptr_x, torch::Tensor ptr_y, 
			 			 float radius, int max_num);