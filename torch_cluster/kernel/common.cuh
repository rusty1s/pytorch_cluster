const int MAX_DIMS = 25;
const int NUM_THREADS = 1024;

inline int GET_BLOCKS(const int n) {
  return (n + NUM_THREADS - 1) / NUM_THREADS;
}

template<typename T>
struct TensorInfo {
  TensorInfo(T *t, int d, int sz[MAX_DIMS], int st[MAX_DIMS]) {
    data = t; dims = d;
    for (int i = 0; i < dims; i++) {
      size[i] = sz[i];
      stride[i] = st[i];
    }
  }

  T *data;
  int dims;
  int size[MAX_DIMS];
  int stride[MAX_DIMS];
};

#define KERNEL_LOOP(I, N) \
  for (int I = blockIdx.x * blockDim.x + threadIdx.x; I < N; i += blockDim.x * gridDim.x)
