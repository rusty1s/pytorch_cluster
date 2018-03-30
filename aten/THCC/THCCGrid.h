void THCCByteGrid(THCudaLongTensor *cluster, THCudaByteTensor *pos, THCudaByteTensor *size,
                  THCudaLongTensor *count);
void THCCCharGrid(THCudaLongTensor *cluster, THCudaCharTensor *pos, THCudaCharTensor *size,
                  THCudaLongTensor *count);
void THCCShortGrid(THCudaLongTensor *cluster, THCudaShortTensor *pos, THCudaShortTensor *size,
                   THCudaLongTensor *count);
void THCCIntGrid(THCudaLongTensor *cluster, THCudaIntTensor *pos, THCudaIntTensor *size,
                 THCudaLongTensor *count);
void THCCLongGrid(THCudaLongTensor *cluster, THCudaLongTensor *pos, THCudaLongTensor *size,
                  THCudaLongTensor *count);
void THCCFloatGrid(THCudaLongTensor *cluster, THCudaTensor *pos, THCudaTensor *size,
                   THCudaLongTensor *count);
void THCCDoubleGrid(THCudaLongTensor *cluster, THCudaDoubleTensor *pos, THCudaDoubleTensor *size,
                    THCudaLongTensor *count);
