#pragma once

#include <torch/extension.h>
#include "utils/neighbors.cpp"
#include <iostream>
#include "compat.h"

torch::Tensor knn_cpu(torch::Tensor support, torch::Tensor query, 
			 int64_t k, int64_t n_threads);

torch::Tensor batch_knn_cpu(torch::Tensor support,
			       torch::Tensor query,
			       torch::Tensor support_batch,
			       torch::Tensor query_batch,
			       int64_t k);