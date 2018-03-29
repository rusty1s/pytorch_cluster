#ifdef __cplusplus
extern "C" {
#endif

int assignColor(THCState *state, THCudaLongTensor *color);

void cluster_serial_kernel(THCState *state, THCudaLongTensor *output, THCudaLongTensor *row, THCudaLongTensor *col, THCudaLongTensor *degree);

void cluster_serial_kernel_Float (THCState *state, THCudaLongTensor *output, THCudaLongTensor *row, THCudaLongTensor *col, THCudaLongTensor *degree,       THCudaTensor *weight);
void cluster_serial_kernel_Double(THCState *state, THCudaLongTensor *output, THCudaLongTensor *row, THCudaLongTensor *col, THCudaLongTensor *degree, THCudaDoubleTensor *weight);
void cluster_serial_kernel_Byte  (THCState *state, THCudaLongTensor *output, THCudaLongTensor *row, THCudaLongTensor *col, THCudaLongTensor *degree,   THCudaByteTensor *weight);
void cluster_serial_kernel_Char  (THCState *state, THCudaLongTensor *output, THCudaLongTensor *row, THCudaLongTensor *col, THCudaLongTensor *degree,   THCudaCharTensor *weight);
void cluster_serial_kernel_Short (THCState *state, THCudaLongTensor *output, THCudaLongTensor *row, THCudaLongTensor *col, THCudaLongTensor *degree,  THCudaShortTensor *weight);
void cluster_serial_kernel_Int   (THCState *state, THCudaLongTensor *output, THCudaLongTensor *row, THCudaLongTensor *col, THCudaLongTensor *degree,    THCudaIntTensor *weight);
void cluster_serial_kernel_Long  (THCState *state, THCudaLongTensor *output, THCudaLongTensor *row, THCudaLongTensor *col, THCudaLongTensor *degree,   THCudaLongTensor *weight);

#ifdef __cplusplus
}
#endif
