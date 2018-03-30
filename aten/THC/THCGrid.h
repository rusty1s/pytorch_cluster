void THCudaByteGrid_cluster(THCudaLongTensor *cluster, THCudaByteTensor *pos,
                            THCudaByteTensor *size, THCudaLongTensor *count);
void THCudaCharGrid_cluster(THCudaLongTensor *cluster, THCudaCharTensor *pos,
                            THCudaCharTensor *size, THCudaLongTensor *count);
void THCudaShortGrid_cluster(THCudaLongTensor *cluster, THCudaShortTensor *pos,
                             THCudaShortTensor *size, THCudaLongTensor *count);
void THCudaIntGrid_cluster(THCudaLongTensor *cluster, THCudaIntTensor *pos,
                           THCudaIntTensor *size, THCudaLongTensor *count);
void THCudaLongGrid_cluster(THCudaLongTensor *cluster, THCudaLongTensor *pos,
                            THCudaLongTensor *size, THCudaLongTensor *count);
void THCudaGrid_cluster(THCudaLongTensor *cluster, THCudaTensor *pos,
                        THCudaTensor *size, THCudaLongTensor *count);
void THCudaDoubleGrid_cluster(THCudaLongTensor *cluster, THCudaDoubleTensor *pos,
                              THCudaDoubleTensor *size, THCudaLongTensor *count);
