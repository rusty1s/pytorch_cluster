void THCudaGreedyWrapper(THCudaLongTensor *cluster,
                         THCudaLongTensor *row,
                         THCudaLongTensor *col,
                         THCudaLongTensor *deg);

void THCudaByteGreedyWrapper(THCudaLongTensor *cluster,
                             THCudaLongTensor *row,
                             THCudaLongTensor *col,
                             THCudaLongTensor *deg,
                             THCudaByteTensor *weight);

void THCudaCharGreedyWrapper(THCudaLongTensor *cluster,
                             THCudaLongTensor *row,
                             THCudaLongTensor *col,
                             THCudaLongTensor *deg,
                             THCudaCharTensor *weight);

void THCudaShortGreedyWrapper(THCudaLongTensor *cluster,
                              THCudaLongTensor *row,
                              THCudaLongTensor *col,
                              THCudaLongTensor *deg,
                              THCudaShortTensor *weight);

void THCudaIntGreedyWrapper(THCudaLongTensor *cluster,
                            THCudaLongTensor *row,
                            THCudaLongTensor *col,
                            THCudaLongTensor *deg,
                            THCudaIntTensor *weight);

void THCudaLongGreedyWrapper(THCudaLongTensor *cluster,
                             THCudaLongTensor *row,
                             THCudaLongTensor *col,
                             THCudaLongTensor *deg,
                             THCudaLongTensor *weight);

void THCudaFloatGreedyWrapper(THCudaLongTensor *cluster,
                              THCudaLongTensor *row,
                              THCudaLongTensor *col,
                              THCudaLongTensor *deg,
                              THCudaTensor *weight);

void THCudaDoubleGreedyWrapper(THCudaLongTensor *cluster,
                               THCudaLongTensor *row,
                               THCudaLongTensor *col,
                               THCudaLongTensor *deg,
                               THCudaDoubleTensor *weight);
