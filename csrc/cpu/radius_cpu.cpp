#include "radius_cpu.h"
#include <algorithm>
#include "utils.h"

torch::Tensor radius_cpu(torch::Tensor query, torch::Tensor support, 
			 float radius, int max_num){

	CHECK_CPU(query);
	CHECK_CPU(support);

	torch::Tensor out;
	std::vector<long> neighbors_indices;
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

	max_count = nanoflann_neighbors<scalar_t>(queries_stl, supports_stl ,neighbors_indices, radius, dim, max_num);

	});

	long* neighbors_indices_ptr = neighbors_indices.data();

	const long long tsize = static_cast<long long>(neighbors_indices.size()/2);
	out = torch::from_blob(neighbors_indices_ptr, {tsize, 2}, options=options);
	out = out.t();

	return out.clone();
}


void get_size_batch(const vector<long>& batch, vector<long>& res){

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

torch::Tensor batch_radius_cpu(torch::Tensor query,
			       torch::Tensor support,
			       torch::Tensor query_batch,
			       torch::Tensor support_batch,
			       float radius, int max_num) {

	torch::Tensor out;
	auto data_qb = query_batch.data_ptr<int64_t>();
	auto data_sb = support_batch.data_ptr<int64_t>();
	std::vector<long> query_batch_stl = std::vector<long>(data_qb, data_qb+query_batch.size(0));
	std::vector<long> size_query_batch_stl;
	get_size_batch(query_batch_stl, size_query_batch_stl);
	std::vector<long> support_batch_stl = std::vector<long>(data_sb, data_sb+support_batch.size(0));
	std::vector<long> size_support_batch_stl;
	get_size_batch(support_batch_stl, size_support_batch_stl);
	std::vector<long> neighbors_indices;
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
							    radius,
								dim,
							    max_num
							    );
	});

	long* neighbors_indices_ptr = neighbors_indices.data();


	const long long tsize = static_cast<long long>(neighbors_indices.size()/2);
	out = torch::from_blob(neighbors_indices_ptr, {tsize, 2}, options=options);
	out = out.t();

	return out.clone();
}