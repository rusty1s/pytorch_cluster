#include "radius_cpu.h"
#include <algorithm>
#include "utils.h"
#include <cstdint>


torch::Tensor knn_cpu(torch::Tensor support, torch::Tensor query, 
			 int64_t k, int64_t n_threads){

	CHECK_CPU(query);
	CHECK_CPU(support);

	torch::Tensor out;
	std::vector<size_t>* neighbors_indices = new std::vector<size_t>(); 
	auto options = torch::TensorOptions().dtype(torch::kLong).device(torch::kCPU);
	int max_count = 0;

	AT_DISPATCH_ALL_TYPES(query.scalar_type(), "radius_cpu", [&] {

	auto data_q = query.data_ptr<scalar_t>();
	auto data_s = support.data_ptr<scalar_t>();
	std::vector<scalar_t> queries_stl = std::vector<scalar_t>(data_q,
								   data_q + query.size(0)*query.size(1));
	std::vector<scalar_t> supports_stl = std::vector<scalar_t>(data_s,
								   data_s + support.size(0)*support.size(1));

	int dim = torch::size(query, 1);

	max_count = nanoflann_neighbors<scalar_t>(queries_stl, supports_stl ,neighbors_indices, 0, dim, 0, n_threads, k, 0);

	});

	size_t* neighbors_indices_ptr = neighbors_indices->data();

	const long long tsize = static_cast<long long>(neighbors_indices->size()/2);
	out = torch::from_blob(neighbors_indices_ptr, {tsize, 2}, options=options);
	out = out.t();

	auto result = torch::zeros_like(out);

	auto index = torch::tensor({1,0});

	result.index_copy_(0, index, out);

	return result;
}


void get_size_batch(const std::vector<long>& batch, std::vector<long>& res){

	res.resize(batch[batch.size()-1]-batch[0]+1, 0);
	long ind = batch[0];
	long incr = 1;
	for(unsigned long i=1; i < batch.size(); i++){

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

torch::Tensor batch_knn_cpu(torch::Tensor support,
			       torch::Tensor query,
			       torch::Tensor support_batch,
			       torch::Tensor query_batch,
			       int64_t k) {

	CHECK_CPU(query);
	CHECK_CPU(support);
	CHECK_CPU(query_batch);
	CHECK_CPU(support_batch);

	torch::Tensor out;
	auto data_qb = query_batch.data_ptr<int64_t>();
	auto data_sb = support_batch.data_ptr<int64_t>();
	
	std::vector<long> query_batch_stl = std::vector<long>(data_qb, data_qb+query_batch.size(0));
	std::vector<long> size_query_batch_stl;
	CHECK_INPUT(std::is_sorted(query_batch_stl.begin(),query_batch_stl.end()));
	get_size_batch(query_batch_stl, size_query_batch_stl);
	
	std::vector<long> support_batch_stl = std::vector<long>(data_sb, data_sb+support_batch.size(0));
	std::vector<long> size_support_batch_stl;
	CHECK_INPUT(std::is_sorted(support_batch_stl.begin(),support_batch_stl.end()));
	get_size_batch(support_batch_stl, size_support_batch_stl);
	
	std::vector<size_t>* neighbors_indices = new std::vector<size_t>(); 
	auto options = torch::TensorOptions().dtype(torch::kLong).device(torch::kCPU);
	int max_count = 0;

	AT_DISPATCH_ALL_TYPES(query.scalar_type(), "batch_radius_cpu", [&] {
	auto data_q = query.data_ptr<scalar_t>();
	auto data_s = support.data_ptr<scalar_t>();
	std::vector<scalar_t> queries_stl = std::vector<scalar_t>(data_q,
								  data_q + query.size(0)*query.size(1));
	std::vector<scalar_t> supports_stl = std::vector<scalar_t>(data_s,
								   data_s + support.size(0)*support.size(1));

	int dim = torch::size(query, 1);
	max_count = batch_nanoflann_neighbors<scalar_t>(queries_stl,
							    supports_stl,
							    size_query_batch_stl,
							    size_support_batch_stl,
							    neighbors_indices,
							    0,
								dim,
							    0,
							    k, 0);
	});

	size_t* neighbors_indices_ptr = neighbors_indices->data();


	const long long tsize = static_cast<long long>(neighbors_indices->size()/2);
	out = torch::from_blob(neighbors_indices_ptr, {tsize, 2}, options=options);
	out = out.t();

	auto result = torch::zeros_like(out);

	auto index = torch::tensor({1,0});

	result.index_copy_(0, index, out);

	return result;
}