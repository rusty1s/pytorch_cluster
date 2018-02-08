int64_t cluster_grid_cuda_Float (THCudaLongTensor *output, THCudaTensor       *position, THCudaTensor       *size, THCudaTensor       *maxPosition);
int64_t cluster_grid_cuda_Double(THCudaLongTensor *output, THCudaDoubleTensor *position, THCudaDoubleTensor *size, THCudaDoubleTensor *maxPosition);
int64_t cluster_grid_cuda_Byte  (THCudaLongTensor *output, THCudaByteTensor   *position, THCudaByteTensor   *size, THCudaByteTensor   *maxPosition);
int64_t cluster_grid_cuda_Char  (THCudaLongTensor *output, THCudaCharTensor   *position, THCudaCharTensor   *size, THCudaCharTensor   *maxPosition);
int64_t cluster_grid_cuda_Short (THCudaLongTensor *output, THCudaShortTensor  *position, THCudaShortTensor  *size, THCudaShortTensor  *maxPosition);
int64_t cluster_grid_cuda_Int   (THCudaLongTensor *output, THCudaIntTensor    *position, THCudaIntTensor    *size, THCudaIntTensor    *maxPosition);
int64_t cluster_grid_cuda_Long  (THCudaLongTensor *output, THCudaLongTensor   *position, THCudaLongTensor   *size, THCudaLongTensor   *maxPosition);
