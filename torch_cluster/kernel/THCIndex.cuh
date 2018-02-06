template <typename a, int Dims>
struct IndexToOffset {
  static __device__ void compute(int i, const TensorInfo<a>& t1, int* t1Offset) {
    int curDimIndex;
    for (int d = Dims - 2; d >= 0; d--) {
      curDimIndex = i % t1.size[d];
      *t1Offset += curDimIndex * t1.stride[d];
      i /= t1.size[d];
    }
  }
};

template <typename a>
struct IndexToOffset<a, -1> {
  static __device__ void compute(int i, const TensorInfo<a>& t1, int* t1Offset) {
    int curDimIndex;
    for (int d = t1.dims - 2; d >= 0; d--) {
      curDimIndex = i % t1.size[d];
      *t1Offset += curDimIndex * t1.stride[d];
      i /= t1.size[d];
    }
  }
};
