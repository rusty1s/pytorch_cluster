#ifdef __cplusplus
extern "C" {
#endif

void cluster_grid_kernel_Float (THCState *state, int C, THCudaLongTensor *output, THCudaTensor       *position, THCudaTensor       *size, THCudaLongTensor *count);
void cluster_grid_kernel_Double(THCState *state, int C, THCudaLongTensor *output, THCudaDoubleTensor *position, THCudaDoubleTensor *size, THCudaLongTensor *count);
void cluster_grid_kernel_Byte  (THCState *state, int C, THCudaLongTensor *output, THCudaByteTensor   *position, THCudaByteTensor   *size, THCudaLongTensor *count);
void cluster_grid_kernel_Char  (THCState *state, int C, THCudaLongTensor *output, THCudaCharTensor   *position, THCudaCharTensor   *size, THCudaLongTensor *count);
void cluster_grid_kernel_Short (THCState *state, int C, THCudaLongTensor *output, THCudaShortTensor  *position, THCudaShortTensor  *size, THCudaLongTensor *count);
void cluster_grid_kernel_Int   (THCState *state, int C, THCudaLongTensor *output, THCudaIntTensor    *position, THCudaIntTensor    *size, THCudaLongTensor *count);
void cluster_grid_kernel_Long  (THCState *state, int C, THCudaLongTensor *output, THCudaLongTensor   *position, THCudaLongTensor   *size, THCudaLongTensor *count);

#ifdef __cplusplus
}
#endif
