#ifdef __cplusplus
extern "C" {
#endif

int64_t cluster_grid_kernel_Float (THCState *state, THCudaLongTensor *output, THCudaTensor       *position, THCudaTensor       *size, THCudaTensor       *maxPosition);
int64_t cluster_grid_kernel_Double(THCState *state, THCudaLongTensor *output, THCudaDoubleTensor *position, THCudaDoubleTensor *size, THCudaDoubleTensor *maxPosition);
int64_t cluster_grid_kernel_Byte  (THCState *state, THCudaLongTensor *output, THCudaByteTensor   *position, THCudaByteTensor   *size, THCudaByteTensor   *maxPosition);
int64_t cluster_grid_kernel_Char  (THCState *state, THCudaLongTensor *output, THCudaCharTensor   *position, THCudaCharTensor   *size, THCudaCharTensor   *maxPosition);
int64_t cluster_grid_kernel_Short (THCState *state, THCudaLongTensor *output, THCudaShortTensor  *position, THCudaShortTensor  *size, THCudaShortTensor  *maxPosition);
int64_t cluster_grid_kernel_Int   (THCState *state, THCudaLongTensor *output, THCudaIntTensor    *position, THCudaIntTensor    *size, THCudaIntTensor    *maxPosition);
int64_t cluster_grid_kernel_Long  (THCState *state, THCudaLongTensor *output, THCudaLongTensor   *position, THCudaLongTensor   *size, THCudaLongTensor   *maxPosition);

#ifdef __cplusplus
}
#endif
