void THGreedy(THLongTensor *cluster, THLongTensor *row, THLongTensor *col, THLongTensor *deg);

void   THByteGreedy(THLongTensor *cluster, THLongTensor *row, THLongTensor *col, THLongTensor *deg,   THByteTensor *weight);
void   THCharGreedy(THLongTensor *cluster, THLongTensor *row, THLongTensor *col, THLongTensor *deg,   THCharTensor *weight);
void  THShortGreedy(THLongTensor *cluster, THLongTensor *row, THLongTensor *col, THLongTensor *deg,  THShortTensor *weight);
void    THIntGreedy(THLongTensor *cluster, THLongTensor *row, THLongTensor *col, THLongTensor *deg,    THIntTensor *weight);
void   THLongGreedy(THLongTensor *cluster, THLongTensor *row, THLongTensor *col, THLongTensor *deg,   THLongTensor *weight);
void  THFloatGreedy(THLongTensor *cluster, THLongTensor *row, THLongTensor *col, THLongTensor *deg,  THFloatTensor *weight);
void THDoubleGreedy(THLongTensor *cluster, THLongTensor *row, THLongTensor *col, THLongTensor *deg, THDoubleTensor *weight);
