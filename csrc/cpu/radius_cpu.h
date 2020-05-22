#pragma once

#include <torch/extension.h>
#include "utils/neighbors.h"
#include "utils/neighbors.cpp"
#include <iostream>
#include "compat.h"

torch::Tensor radius_cpu(torch::Tensor query, torch::Tensor support,
			 			 double radius, int64_t max_num);

torch::Tensor batch_radius_cpu(torch::Tensor query,
			       torch::Tensor support,
			       torch::Tensor query_batch,
			       torch::Tensor support_batch,
			       double radius, int64_t max_num);