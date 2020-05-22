// Author: Peiyuan Liao (alexander_liao@outlook.com)
//


# pragma once

#include <ATen/ATen.h>
#include <cmath>
#include <vector>
#include <unordered_map>
#include <map>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <iomanip>

#include <time.h>


template<typename scalar_t>
struct PointCloud
{
	std::vector<std::vector<scalar_t>*> pts;

	void set(std::vector<scalar_t> new_pts, int dim){

		std::vector<std::vector<scalar_t>*> temp(new_pts.size()/dim);
		for(size_t i=0; i < new_pts.size(); i++){
			if(i%dim == 0){
				std::vector<scalar_t>* point = new std::vector<scalar_t>(dim);

				for (size_t j = 0; j < (size_t)dim; j++) {
					(*point)[j]=new_pts[i+j];
				}
				temp[i/dim] = point;
			}
		}

		pts = temp;
	}
	void set_batch(std::vector<scalar_t> new_pts, size_t begin, long size, int dim){
		std::vector<std::vector<scalar_t>*> temp(size);
		for(size_t i=0; i < (size_t)size; i++){
			std::vector<scalar_t>* point = new std::vector<scalar_t>(dim);
			for (size_t j = 0; j < (size_t)dim; j++) {
					(*point)[j] = new_pts[dim*(begin+i)+j];
			}

			temp[i] = point;

		}
		pts = temp;
	}

	// Must return the number of data points
	inline size_t kdtree_get_point_count() const { return pts.size(); }

	// Returns the dim'th component of the idx'th point in the class:
	inline scalar_t kdtree_get_pt(const size_t idx, const size_t dim) const
	{
		return (*pts[idx])[dim];
	}

	// Optional bounding-box computation: return false to default to a standard bbox computation loop.
	//   Return true if the BBOX was already computed by the class and returned in "bb" so it can be avoided to redo it again.
	//   Look at bb.size() to find out the expected dimensionality (e.g. 2 or 3 for point clouds)
	template <class BBOX>
	bool kdtree_get_bbox(BBOX& /* bb */) const { return false; }


};
