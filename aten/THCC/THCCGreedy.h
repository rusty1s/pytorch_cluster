void THCCGreedy(THCudaLongTensor *cluster, THCudaLongTensor *row, THCudaLongTensor *col);

void   THCCByteGreedy(THCudaLongTensor *cluster, THCudaLongTensor *row, THCudaLongTensor *col,   THCudaByteTensor *weight);
void   THCCCharGreedy(THCudaLongTensor *cluster, THCudaLongTensor *row, THCudaLongTensor *col,   THCudaCharTensor *weight);
void  THCCShortGreedy(THCudaLongTensor *cluster, THCudaLongTensor *row, THCudaLongTensor *col,  THCudaShortTensor *weight);
void    THCCIntGreedy(THCudaLongTensor *cluster, THCudaLongTensor *row, THCudaLongTensor *col,    THCudaIntTensor *weight);
void   THCCLongGreedy(THCudaLongTensor *cluster, THCudaLongTensor *row, THCudaLongTensor *col,   THCudaLongTensor *weight);
void  THCCFloatGreedy(THCudaLongTensor *cluster, THCudaLongTensor *row, THCudaLongTensor *col,       THCudaTensor *weight);
void THCCDoubleGreedy(THCudaLongTensor *cluster, THCudaLongTensor *row, THCudaLongTensor *col, THCudaDoubleTensor *weight);
