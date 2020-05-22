
// 3D Version https://github.com/HuguesTHOMAS/KPConv

#include "neighbors.h"

template<typename scalar_t>
size_t nanoflann_neighbors(vector<scalar_t>& queries, vector<scalar_t>& supports,
			vector<size_t>*& neighbors_indices, double radius, int dim, int64_t max_num){

	const scalar_t search_radius = static_cast<scalar_t>(radius*radius);

	// Counting vector
	size_t max_count = 1;

	// CLoud variable
	PointCloud<scalar_t> pcd;
	pcd.set(supports, dim);
	//Cloud query
	PointCloud<scalar_t> pcd_query;
	pcd_query.set(queries, dim);

	// Tree parameters
	nanoflann::KDTreeSingleIndexAdaptorParams tree_params(15 /* max leaf */);

	// KDTree type definition
	typedef nanoflann::KDTreeSingleIndexAdaptor< nanoflann::L2_Adaptor<scalar_t, PointCloud<scalar_t> > , PointCloud<scalar_t>> my_kd_tree_t;

	// Pointer to trees
	my_kd_tree_t* index;
	index = new my_kd_tree_t(dim, pcd, tree_params);
	index->buildIndex();
	// Search neigbors indices
	// ***********************

	// Search params
	nanoflann::SearchParams search_params;
	// search_params.sorted = true;
	std::vector< std::vector<std::pair<size_t, scalar_t> > > list_matches(pcd_query.pts.size());

	double eps = 0.000001;

	// indices
	size_t i0 = 0;

	for (auto& p : pcd_query.pts){
		auto p0 = *p;
		// Find neighbors
		scalar_t* query_pt = new scalar_t[dim];
		std::copy(p0.begin(), p0.end(), query_pt); 

		list_matches[i0].reserve(max_count);
		std::vector<std::pair<size_t, scalar_t> > ret_matches;

		const size_t nMatches = index->radiusSearch(query_pt, (scalar_t)(search_radius+eps), ret_matches, search_params);
		
		list_matches[i0] = ret_matches;
		if(max_count < nMatches) max_count = nMatches;
		i0++;

	}
	// Reserve the memory
	if(max_num > 0) {
		max_count = max_num;
	}
	
	size_t size = 0; // total number of edges
	for (auto& inds : list_matches){
		if(inds.size() <= max_count)
			size += inds.size();
		else
			size += max_count;
	}

	neighbors_indices->resize(size*2);
	size_t i1 = 0; // index of the query points
	size_t u = 0; // curent index of the neighbors_indices
	for (auto& inds : list_matches){
		for (size_t j = 0; j < max_count; j++){
			if(j < inds.size()){
				(*neighbors_indices)[u] = inds[j].first;
				(*neighbors_indices)[u + 1] = i1;
				u += 2;
			}
		}
		i1++;
	}

	return max_count;




}

template<typename scalar_t>
size_t batch_nanoflann_neighbors (vector<scalar_t>& queries,
                               vector<scalar_t>& supports,
                               vector<long>& q_batches,
                               vector<long>& s_batches,
                               vector<size_t>*& neighbors_indices,
                               double radius, int dim, int64_t max_num){


// Initiate variables
// ******************
// indices
	size_t i0 = 0;

// Square radius
	const scalar_t r2 = static_cast<scalar_t>(radius*radius);

	// Counting vector
	size_t max_count = 0;

	// batch index
	size_t b = 0;
	size_t sum_qb = 0;
	size_t sum_sb = 0;

	double eps = 0.000001;
	// Nanoflann related variables
	// ***************************

	// CLoud variable
	PointCloud<scalar_t> current_cloud;
	PointCloud<scalar_t> query_pcd;
	query_pcd.set(queries, dim);
	vector<vector<pair<size_t, scalar_t> > > all_inds_dists(query_pcd.pts.size());

	// Tree parameters
	nanoflann::KDTreeSingleIndexAdaptorParams tree_params(10 /* max leaf */);

	// KDTree type definition
	typedef nanoflann::KDTreeSingleIndexAdaptor< nanoflann::L2_Adaptor<scalar_t, PointCloud<scalar_t> > , PointCloud<scalar_t>> my_kd_tree_t;

// Pointer to trees
	my_kd_tree_t* index;
    // Build KDTree for the first batch element
	current_cloud.set_batch(supports, sum_sb, s_batches[b], dim);
	index = new my_kd_tree_t(dim, current_cloud, tree_params);
	index->buildIndex();
// Search neigbors indices
// ***********************
// Search params
	nanoflann::SearchParams search_params;
	search_params.sorted = true;

	for (auto& p : query_pcd.pts){
		auto p0 = *p;
// Check if we changed batch

		scalar_t* query_pt = new scalar_t[dim];
		std::copy(p0.begin(), p0.end(), query_pt); 

		if (i0 == sum_qb + q_batches[b]){
			sum_qb += q_batches[b];
			sum_sb += s_batches[b];
			b++;

// Change the points
			current_cloud.pts.clear();
			current_cloud.set_batch(supports, sum_sb, s_batches[b], dim);
// Build KDTree of the current element of the batch
			delete index;
			index = new my_kd_tree_t(dim, current_cloud, tree_params);
			index->buildIndex();
		}
// Initial guess of neighbors size
		all_inds_dists[i0].reserve(max_count);
// Find neighbors
		size_t nMatches = index->radiusSearch(query_pt, r2+eps, all_inds_dists[i0], search_params);
// Update max count

		std::vector<std::pair<size_t, float> > indices_dists;
		nanoflann::RadiusResultSet<float,size_t> resultSet(r2, indices_dists);

		index->findNeighbors(resultSet, query_pt, search_params);

		if (nMatches > max_count)
			max_count = nMatches;
// Increment query idx
		i0++;
	}
	// how many neighbors do we keep
	if(max_num > 0) {
		max_count = max_num;
	}
// Reserve the memory
	
	size_t size = 0; // total number of edges
	for (auto& inds_dists : all_inds_dists){
		if(inds_dists.size() <= max_count)
			size += inds_dists.size();
		else
			size += max_count;
	}
	neighbors_indices->resize(size * 2);
	i0 = 0;
	sum_sb = 0;
	sum_qb = 0;
	b = 0;
	size_t u = 0;
	for (auto& inds_dists : all_inds_dists){
		if (i0 == sum_qb + q_batches[b]){
			sum_qb += q_batches[b];
			sum_sb += s_batches[b];
			b++;
		}
		for (size_t j = 0; j < max_count; j++){
			if (j < inds_dists.size()){
				(*neighbors_indices)[u] = inds_dists[j].first + sum_sb;
				(*neighbors_indices)[u + 1] = i0;
				u += 2;
			}
		}
		i0++;
	}
	
	return max_count;
}
