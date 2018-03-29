void THGreedy_cluster(THLongTensor *cluster, THLongTensor *row, THLongTensor *col,
                      THLongTensor *deg);
void THByteGreedy_cluster(THLongTensor *cluster, THLongTensor *row, THLongTensor *col,
                          THLongTensor *deg, THByteTensor *weight);
void THCharGreedy_cluster(THLongTensor *cluster, THLongTensor *row, THLongTensor *col,
                          THLongTensor *deg, THCharTensor *weight);
void THShortGreedy_cluster(THLongTensor *cluster, THLongTensor *row, THLongTensor *col,
                           THLongTensor *deg, THShortTensor *weight);
void THIntGreedy_cluster(THLongTensor *cluster, THLongTensor *row, THLongTensor *col,
                         THLongTensor *deg, THIntTensor *weight);
void THLongGreedy_cluster(THLongTensor *cluster, THLongTensor *row, THLongTensor *col,
                          THLongTensor *deg, THLongTensor *weight);
void THFloatGreedy_cluster(THLongTensor *cluster, THLongTensor *row, THLongTensor *col,
                           THLongTensor *deg, THFloatTensor *weight);
void THDoubleGreedy_cluster(THLongTensor *cluster, THLongTensor *row, THLongTensor *col,
                            THLongTensor *deg, THDoubleTensor *weight);
