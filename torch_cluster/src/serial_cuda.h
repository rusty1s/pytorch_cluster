void cluster_serial_cuda(THCudaLongTensor *output, THCudaLongTensor *row, THCudaLongTensor *col, THCudaLongTensor *degree);

void cluster_serial_cuda_Float (THCudaLongTensor *output, THCudaLongTensor *row, THCudaLongTensor *col,       THCudaTensor *weight, THCudaLongTensor *degree);
void cluster_serial_cuda_Double(THCudaLongTensor *output, THCudaLongTensor *row, THCudaLongTensor *col, THCudaDoubleTensor *weight, THCudaLongTensor *degree);
void cluster_serial_cuda_Byte  (THCudaLongTensor *output, THCudaLongTensor *row, THCudaLongTensor *col,   THCudaByteTensor *weight, THCudaLongTensor *degree);
void cluster_serial_cuda_Char  (THCudaLongTensor *output, THCudaLongTensor *row, THCudaLongTensor *col,   THCudaCharTensor *weight, THCudaLongTensor *degree);
void cluster_serial_cuda_Short (THCudaLongTensor *output, THCudaLongTensor *row, THCudaLongTensor *col,  THCudaShortTensor *weight, THCudaLongTensor *degree);
void cluster_serial_cuda_Int   (THCudaLongTensor *output, THCudaLongTensor *row, THCudaLongTensor *col,    THCudaIntTensor *weight, THCudaLongTensor *degree);
void cluster_serial_cuda_Long  (THCudaLongTensor *output, THCudaLongTensor *row, THCudaLongTensor *col,   THCudaLongTensor *weight, THCudaLongTensor *degree);
