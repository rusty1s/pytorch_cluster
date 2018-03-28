#ifdef __cplusplus
extern "C" {
#endif

void cluster_serial_kernel(THCState *state, THCudaLongTensor *output, THCudaLongTensor *row, THCudaLongTensor *col, THCudaLongTensor *degree);

void cluster_serial_kernel_Float (THCState *state, THCudaLongTensor *output, THCudaLongTensor *row, THCudaLongTensor *col,       THCudaTensor *weight, THCudaLongTensor *degree);
void cluster_serial_kernel_Double(THCState *state, THCudaLongTensor *output, THCudaLongTensor *row, THCudaLongTensor *col, THCudaDoubleTensor *weight, THCudaLongTensor *degree);
void cluster_serial_kernel_Byte  (THCState *state, THCudaLongTensor *output, THCudaLongTensor *row, THCudaLongTensor *col,   THCudaByteTensor *weight, THCudaLongTensor *degree);
void cluster_serial_kernel_Char  (THCState *state, THCudaLongTensor *output, THCudaLongTensor *row, THCudaLongTensor *col,   THCudaCharTensor *weight, THCudaLongTensor *degree);
void cluster_serial_kernel_Short (THCState *state, THCudaLongTensor *output, THCudaLongTensor *row, THCudaLongTensor *col,  THCudaShortTensor *weight, THCudaLongTensor *degree);
void cluster_serial_kernel_Int   (THCState *state, THCudaLongTensor *output, THCudaLongTensor *row, THCudaLongTensor *col,    THCudaIntTensor *weight, THCudaLongTensor *degree);
void cluster_serial_kernel_Long  (THCState *state, THCudaLongTensor *output, THCudaLongTensor *row, THCudaLongTensor *col,   THCudaLongTensor *weight, THCudaLongTensor *degree);

#ifdef __cplusplus
}
#endif
