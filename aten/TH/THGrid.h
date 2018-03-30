void THByteGrid_cluster(THLongTensor *cluster,
                        THByteTensor *pos,
                        THByteTensor *size,
                        THLongTensor *count);

void THCharGrid_cluster(THLongTensor *cluster,
                        THCharTensor *pos,
                        THCharTensor *size,
                        THLongTensor *count);

void THShortGrid_cluster(THLongTensor *cluster,
                         THShortTensor *pos,
                         THShortTensor *size,
                         THLongTensor *count);

void THIntGrid_cluster(THLongTensor *cluster,
                       THIntTensor *pos,
                       THIntTensor *size,
                       THLongTensor *count);

void THLongGrid_cluster(THLongTensor *cluster,
                        THLongTensor *pos,
                        THLongTensor *size,
                        THLongTensor *count);

void THFloatGrid_cluster(THLongTensor *cluster,
                         THFloatTensor *pos,
                         THFloatTensor *size,
                         THLongTensor *count);

void THDoubleGrid_cluster(THLongTensor *cluster,
                          THDoubleTensor *pos,
                          THDoubleTensor *size,
                          THLongTensor *count);
