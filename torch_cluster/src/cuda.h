void cluster_grid_cuda_Float (int C, THCudaLongTensor *output, THCudaTensor       *position, THCudaTensor       *size, THCudaLongTensor *count);
void cluster_grid_cuda_Double(int C, THCudaLongTensor *output, THCudaDoubleTensor *position, THCudaDoubleTensor *size, THCudaLongTensor *count);
void cluster_grid_cuda_Byte  (int C, THCudaLongTensor *output, THCudaByteTensor   *position, THCudaByteTensor   *size, THCudaLongTensor *count);
void cluster_grid_cuda_Char  (int C, THCudaLongTensor *output, THCudaCharTensor   *position, THCudaCharTensor   *size, THCudaLongTensor *count);
void cluster_grid_cuda_Short (int C, THCudaLongTensor *output, THCudaShortTensor  *position, THCudaShortTensor  *size, THCudaLongTensor *count);
void cluster_grid_cuda_Int   (int C, THCudaLongTensor *output, THCudaIntTensor    *position, THCudaIntTensor    *size, THCudaLongTensor *count);
void cluster_grid_cuda_Long  (int C, THCudaLongTensor *output, THCudaLongTensor   *position, THCudaLongTensor   *size, THCudaLongTensor *count);
