
#include <torch/extension.h>
#include <ATen/ATen.h>
#include "utils/neighbors.h"
#include "utils/neighbors.cpp"
#include <iostream>
#include "compat.h"

at::Tensor radius_search(at::Tensor query,
			 at::Tensor support,
			 float radius, int max_num=-1, int mode=0){

	at::Tensor out;
	std::vector<long> neighbors_indices;
	auto options = torch::TensorOptions().dtype(torch::kLong).device(torch::kCPU);
	int max_count = 0;
	AT_DISPATCH_ALL_TYPES(query.scalar_type(), "radius_search", [&] {

	auto data_q = query.DATA_PTR<scalar_t>();
	auto data_s = support.DATA_PTR<scalar_t>();
	std::vector<scalar_t> queries_stl = std::vector<scalar_t>(data_q,
								   data_q + query.size(0)*query.size(1));
	std::vector<scalar_t> supports_stl = std::vector<scalar_t>(data_s,
								   data_s + support.size(0)*support.size(1));

	max_count = nanoflann_neighbors<scalar_t>(queries_stl, supports_stl ,neighbors_indices, radius, max_num, mode);

	});

	long* neighbors_indices_ptr = neighbors_indices.data();
	if(mode == 0)
		out = torch::from_blob(neighbors_indices_ptr, {query.size(0), max_count}, options=options);
	else if(mode ==1)
		out = torch::from_blob(neighbors_indices_ptr, {neighbors_indices.size()/2, 2}, options=options);

	return out.clone();
}

void get_size_batch(const vector<long>& batch, vector<long>& res){

	res.resize(batch[batch.size()-1]-batch[0]+1, 0);
	long ind = batch[0];
	long incr = 1;
	for(int i=1; i < batch.size(); i++){

		if(batch[i] == ind)
			incr++;
		else{
			res[ind-batch[0]] = incr;
			incr =1;
			ind = batch[i];
		}
	}
	res[ind-batch[0]] = incr;
}

at::Tensor batch_radius_search(at::Tensor query,
			       at::Tensor support,
			       at::Tensor query_batch,
			       at::Tensor support_batch,
			       float radius, int max_num=-1, int mode=0) {
	at::Tensor out;
	auto data_qb = query_batch.data_ptr<long>();
	auto data_sb = support_batch.data_ptr<long>();
	std::vector<long> query_batch_stl = std::vector<long>(data_qb, data_qb+query_batch.size(0));
	std::vector<long> size_query_batch_stl;
	get_size_batch(query_batch_stl, size_query_batch_stl);
	std::vector<long> support_batch_stl = std::vector<long>(data_sb, data_sb+support_batch.size(0));
	std::vector<long> size_support_batch_stl;
	get_size_batch(support_batch_stl, size_support_batch_stl);
	std::vector<long> neighbors_indices;
	auto options = torch::TensorOptions().dtype(torch::kLong).device(torch::kCPU);
	int max_count = 0;
	AT_DISPATCH_ALL_TYPES(query.scalar_type(), "batch_radius_search", [&] {
	auto data_q = query.data_ptr<scalar_t>();
	auto data_s = support.data_ptr<scalar_t>();
	std::vector<scalar_t> queries_stl = std::vector<scalar_t>(data_q,
								  data_q + query.size(0)*query.size(1));
	std::vector<scalar_t> supports_stl = std::vector<scalar_t>(data_s,
								   data_s + support.size(0)*support.size(1));
	max_count = batch_nanoflann_neighbors<scalar_t>(queries_stl,
							    supports_stl,
							    size_query_batch_stl,
							    size_support_batch_stl,
							    neighbors_indices,
							    radius,
							    max_num,
							    mode);
	});
	long* neighbors_indices_ptr = neighbors_indices.data();


	if(mode == 0)
		out = torch::from_blob(neighbors_indices_ptr, {query.size(0), max_count}, options=options);
	else if(mode == 1)
		out = torch::from_blob(neighbors_indices_ptr, {neighbors_indices.size()/2, 2}, options=options);
	return out.clone();
}
using namespace pybind11::literals;
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("radius_search",
	      &radius_search,
	      "compute the radius search of a point cloud using nanoflann"
	      "-query : a pytorch tensor of size N1 x 3,. used to query the nearest neighbors"
	      "- support : a pytorch tensor of size N2 x 3. used to build the tree"
	      "-  radius : float number, size of the ball for the radius search."
	      "- max_num : int number, indicate the maximum of neaghbors allowed(if -1 then all the possible neighbors will be computed). "
	      " - mode : int number that indicate which format for the neighborhood"
	      " mode=0 mean a matrix of neighbors(-1 for shadow neighbors)"
	      "mode=1 means a matrix of edges of size Num_edge x 2"
	      "return a tensor of size N1 x M where M is either max_num or the maximum number of neighbors found if mode = 0, if mode=1 return a tensor of size Num_edge x 2.",
	      "query"_a, "support"_a, "radius"_a, "max_num"_a=-1, "mode"_a=0);
	m.def("batch_radius_search",
	      &batch_radius_search,
		"compute the radius search of a point cloud for each batch using nanoflann"
		"-query : a pytorch tensor (float) of size N1 x 3,. used to query the nearest neighbors"
		"- support : a pytorch tensor(float) of size N2 x 3. used to build the tree"
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
	      "query"_a, "support"_a, "query_batch"_a, "support_batch"_a, "radius"_a, "max_num"_a=-1, "mode"_a=0);
}
