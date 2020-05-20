

#include "cloud.h"
#include "nanoflann.hpp"
#include <set>
#include <cstdint>

using namespace std;


template<typename scalar_t>
int nanoflann_neighbors(vector<scalar_t>& queries, vector<scalar_t>& supports,
			vector<long>& neighbors_indices, float radius, int dim, int max_num, int mode);