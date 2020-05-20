#include "radius_cpu.h"

#include "utils.h"

torch::Tensor radius_cpu(torch::Tensor query, torch::Tensor support, 
             torch::Tensor ptr_x, torch::Tensor ptr_y, 
			 float radius, int max_num){

	CHECK_CPU(query);
	CHECK_CPU(support);

	/*
	x = torch.cat([x, 2 * r * batch_x.view(-1, 1).to(x.dtype)], dim=-1)
    y = torch.cat([y, 2 * r * batch_y.view(-1, 1).to(y.dtype)], dim=-1)
	auto batch_x = ptr_x.clone();
	auto batch_y = ptr_y.clone();

	batch_x._mul(2*radius);
	batch_y._mul(2*radius);

	auto query = torch::cat({query,batch_x},-1);
	auto support = torch::cat({support,batch_y},-1);
	*/

	torch::Tensor out;
	std::vector<long> neighbors_indices;
	auto options = torch::TensorOptions().dtype(torch::kLong).device(torch::kCPU);
	int max_count = 0;

	AT_DISPATCH_ALL_TYPES(query.scalar_type(), "radius_cpu", [&] {

	auto data_q = query.DATA_PTR<scalar_t>();
	auto data_s = support.DATA_PTR<scalar_t>();
	std::vector<scalar_t> queries_stl = std::vector<scalar_t>(data_q,
								   data_q + query.size(0)*query.size(1));
	std::vector<scalar_t> supports_stl = std::vector<scalar_t>(data_s,
								   data_s + support.size(0)*support.size(1));

	int dim = torch::size(query, 1);

	max_count = nanoflann_neighbors<scalar_t>(queries_stl, supports_stl ,neighbors_indices, radius, dim, max_num);

	});

	long* neighbors_indices_ptr = neighbors_indices.data();
	out = torch::from_blob(neighbors_indices_ptr, {neighbors_indices.size()/2, 2}, options=options);

	return out.t().clone();
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