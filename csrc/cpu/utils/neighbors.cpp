
// 3D Version https://github.com/HuguesTHOMAS/KPConv

#include "neighbors.h"

template<typename scalar_t>
int nanoflann_neighbors(vector<scalar_t>& queries, vector<scalar_t>& supports,
			vector<long>& neighbors_indices, float radius, int dim, int max_num){

	// Initiate variables
	// ******************

	const scalar_t search_radius = static_cast<scalar_t>(radius*radius);

	// Counting vector
	size_t max_count = 1;

	// Nanoflann related variables
	// ***************************

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
	search_params.sorted = true;
	std::vector< std::vector<std::pair<size_t, scalar_t> > > list_matches(pcd_query.pts.size());

	float eps = 0.00001;

	// indices
	size_t i0 = 0;

	for (auto& p0 : pcd_query.pts){

		// Find neighbors
		scalar_t* query_pt = new scalar_t[dim];
		std::copy(p0.begin(), p0.end(), query_pt); 

		//for(int i=0; i < p0.size(); i++)
		//std::cout << query_pt[i] << '\n';

		list_matches[i0].reserve(max_count);
		std::vector<std::pair<size_t, scalar_t> > ret_matches;

		const size_t nMatches = index->radiusSearch(query_pt, search_radius+eps, ret_matches, search_params);
		list_matches[i0] = ret_matches;
		if(max_count < nMatches) max_count = nMatches;
		i0++;


		// Get worst (furthest) point, without sorting:
		
		// cout << "\n neighbors: " << nMatches << "\n";

		// Get worst (furthest) point, without sorting:
		// for(int i=0; i < ret_matches.size(); i++)
		// std::cout << ret_matches.at(i) << '\n';

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

	neighbors_indices.resize(size*2);
	size_t i1 = 0; // index of the query points
	size_t u = 0; // curent index of the neighbors_indices
	for (auto& inds : list_matches){
		for (size_t j = 0; j < max_count; j++){
			if(j < inds.size()){
				neighbors_indices[u] = inds[j].first;
				neighbors_indices[u + 1] = i1;
				u += 2;
			}
		}
		i1++;
	}

	return max_count;




}