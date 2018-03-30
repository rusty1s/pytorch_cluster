void THCGreedy_cluster(THCudaLongTensor *cluster, THCudaLongTensor *row,
                       THCudaLongTensor *col, THCudaLongTensor *deg);
void THCudaByteGreedy_cluster(THCudaLongTensor *cluster, THCudaLongTensor *row,
                              THCudaLongTensor *col, THCudaLongTensor *deg,
                              THCudaByteTensor *weight);
void THCudaCharGreedy_cluster(THCudaLongTensor *cluster, THCudaLongTensor *row,
                              THCudaLongTensor *col, THCudaLongTensor *deg,
                              THCudaCharTensor *weight);
void THCudaShortGreedy_cluster(THCudaLongTensor *cluster, THCudaLongTensor *row,
                               THCudaLongTensor *col, THCudaLongTensor *deg,
                               THCudaShortTensor *weight);
void THCudaIntGreedy_cluster(THCudaLongTensor *cluster, THCudaLongTensor *row,
                             THCudaLongTensor *col, THCudaLongTensor *deg,
                             THCudaIntTensor *weight);
void THCudaLongGreedy_cluster(THCudaLongTensor *cluster, THCudaLongTensor *row,
                              THCudaLongTensor *col, THCudaLongTensor *deg,
                              THCudaLongTensor *weight);
void THCudaGreedy_cluster(THCudaLongTensor *cluster, THCudaLongTensor *row,
                          THCudaLongTensor *col, THCudaLongTensor *deg,
                          THCudaTensor *weight);
void THCudaDoubleGreedy_cluster(THCudaLongTensor *cluster, THCudaLongTensor *row,
                                THCudaLongTensor *col, THCudaLongTensor *deg,
                                THCudaDoubleTensor *weight);
