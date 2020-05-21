

#include "cloud.h"
#include "nanoflann.hpp"
#include <set>
#include <cstdint>
#include <thread>

using namespace std;


template<typename scalar_t>
int nanoflann_neighbors(vector<scalar_t>& queries, vector<scalar_t>& supports,
			vector<long>& neighbors_indices, float radius, int dim, int max_num);

template<typename scalar_t>
int batch_nanoflann_neighbors (vector<scalar_t>& queries,
                               vector<scalar_t>& supports,
                               vector<long>& q_batches,
                               vector<long>& s_batches,
                               vector<long>& neighbors_indices,
                               float radius, int dim, int max_num);