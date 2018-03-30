void THCCGreedy(THCudaLongTensor *cluster, THCudaLongTensor *row, THCudaLongTensor *col,
                THCudaLongTensor *deg);

void THCCByteGreedy(THCudaLongTensor *cluster, THCudaLongTensor *row, THCudaLongTensor *col,
                    THCudaLongTensor *deg, THCudaByteTensor *weight);
void THCCCharGreedy(THCudaLongTensor *cluster, THCudaLongTensor *row, THCudaLongTensor *col,
                    THCudaLongTensor *deg, THCudaCharTensor *weight);
void THCCShortGreedy(THCudaLongTensor *cluster, THCudaLongTensor *row, THCudaLongTensor *col,
                     THCudaLongTensor *deg, THCudaShortTensor *weight);
void THCCIntGreedy(THCudaLongTensor *cluster, THCudaLongTensor *row, THCudaLongTensor *col,
                   THCudaLongTensor *deg, THCudaIntTensor *weight);
void THCCLongGreedy(THCudaLongTensor *cluster, THCudaLongTensor *row, THCudaLongTensor *col,
                    THCudaLongTensor *deg, THCudaLongTensor *weight);
void THCCFloatGreedy(THCudaLongTensor *cluster, THCudaLongTensor *row, THCudaLongTensor *col,
                     THCudaLongTensor *deg, THCudaTensor *weight);
void THCCDoubleGreedy(THCudaLongTensor *cluster, THCudaLongTensor *row, THCudaLongTensor *col,
                      THCudaLongTensor *deg, THCudaDoubleTensor *weight);
