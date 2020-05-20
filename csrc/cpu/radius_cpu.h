#pragma once

#include <torch/extension.h>
#include <ATen/ATen.h>
#include "utils/neighbors.h"
#include "utils/neighbors.cpp"
#include <iostream>
#include "compat.h"

torch::Tensor radius_cpu(torch::Tensor query,
			 torch::Tensor support,torch::Tensor ptr_x,
                     torch::Tensor ptr_y, 
			 float radius, int max_num);
/*
using namespace pybind11::literals;
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("radius_search",
	      &radius_search,
	      "compute the radius search of a point cloud using nanoflann"
	      "-query : a pytorch tensor of size N1 x d,. used to query the nearest neighbors"
	      "- support : a pytorch tensor of size N2 x d. used to build the tree"
	      "-  radius : float number, size of the ball for the radius search."
	      "- max_num : int number, indicate the maximum of neaghbors allowed(if -1 then all the possible neighbors will be computed). "
	      " - mode : int number that indicate which format for the neighborhood"
	      " mode=0 mean a matrix of neighbors(-1 for shadow neighbors)"
	      "mode=1 means a matrix of edges of size Num_edge x 2"
	      "return a tensor of size N1 x M where M is either max_num or the maximum number of neighbors found if mode = 0, if mode=1 return a tensor of size Num_edge x 2.",
	      "query"_a, "support"_a, "radius"_a, "dim"_a, "max_num"_a=-1, "mode"_a=0);
	m.def("batch_radius_search",
	      &batch_radius_search,
		"compute the radius search of a point cloud for each batch using nanoflann"
		"-query : a pytorch tensor (float) of size N1 x d,. used to query the nearest neighbors"
		"- support : a pytorch tensor(float) of size N2 x d. used to build the tree"
		"- query_batch : a pytorch tensor(long) contains indices of the batch of the query size N1"
	      "NB : the batch must be sorted"
		"- support_batch: a pytorch tensor(long) contains indices of the batch of the support size N2"
	      "NB: the batch must be sorted"
		"-radius: float number, size of the ball for the radius search."
		"- max_num : int number, indicate the maximum of neaghbors allowed(if -1 then all the possible neighbors wrt the radius will be computed)."
		"- mode : int number that indicate which format for the neighborhood"
		"mode=0 mean a matrix of neighbors(N2 for shadow neighbors)"
		"mode=1 means a matrix of edges of size Num_edge x 2"
	      "return a tensor of size N1 x M where M is either max_num or the maximum number of neighbors found if mode = 0, if mode=1 return a tensor of size Num_edge x 2.",
	      "query"_a, "support"_a, "query_batch"_a, "support_batch"_a, "radius"_a, "dim"_a, "max_num"_a=-1, "mode"_a=0);
}
*/