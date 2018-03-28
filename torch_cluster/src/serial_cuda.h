void cluster_serial_cuda(THCudaLongTensor *output, THCudaLongTensor *row, THCudaLongTensor *col, THCudaLongTensor *degree);

void cluster_serial_cuda_Float (THCudaLongTensor *output, THCudaLongTensor *row, THCudaLongTensor *col, THCudaLongTensor *degree,       THCudaTensor *weight);
void cluster_serial_cuda_Double(THCudaLongTensor *output, THCudaLongTensor *row, THCudaLongTensor *col, THCudaLongTensor *degree, THCudaDoubleTensor *weight);
void cluster_serial_cuda_Byte  (THCudaLongTensor *output, THCudaLongTensor *row, THCudaLongTensor *col, THCudaLongTensor *degree,   THCudaByteTensor *weight);
void cluster_serial_cuda_Char  (THCudaLongTensor *output, THCudaLongTensor *row, THCudaLongTensor *col, THCudaLongTensor *degree,   THCudaCharTensor *weight);
void cluster_serial_cuda_Short (THCudaLongTensor *output, THCudaLongTensor *row, THCudaLongTensor *col, THCudaLongTensor *degree,  THCudaShortTensor *weight);
void cluster_serial_cuda_Int   (THCudaLongTensor *output, THCudaLongTensor *row, THCudaLongTensor *col, THCudaLongTensor *degree,    THCudaIntTensor *weight);
void cluster_serial_cuda_Long  (THCudaLongTensor *output, THCudaLongTensor *row, THCudaLongTensor *col, THCudaLongTensor *degree,   THCudaLongTensor *weight);
