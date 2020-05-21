#pragma once

#include <torch/extension.h>
#include "utils/neighbors.h"
#include "utils/neighbors.cpp"
#include <iostream>
#include "compat.h"

torch::Tensor radius_cpu(torch::Tensor query, torch::Tensor support,
			 			 float radius, int max_num);

torch::Tensor batch_radius_cpu(torch::Tensor query,
			       torch::Tensor support,
			       torch::Tensor query_batch,
			       torch::Tensor support_batch,
			       float radius, int max_num);