void THGreedy(THLongTensor *cluster, THLongTensor *row, THLongTensor *col);

void   THByteGreedy(THLongTensor *cluster, THLongTensor *row, THLongTensor *col,   THByteTensor *weight);
void   THCharGreedy(THLongTensor *cluster, THLongTensor *row, THLongTensor *col,   THCharTensor *weight);
void  THShortGreedy(THLongTensor *cluster, THLongTensor *row, THLongTensor *col,  THShortTensor *weight);
void    THIntGreedy(THLongTensor *cluster, THLongTensor *row, THLongTensor *col,    THIntTensor *weight);
void   THLongGreedy(THLongTensor *cluster, THLongTensor *row, THLongTensor *col,   THLongTensor *weight);
void  THFloatGreedy(THLongTensor *cluster, THLongTensor *row, THLongTensor *col,  THFloatTensor *weight);
void THDoubleGreedy(THLongTensor *cluster, THLongTensor *row, THLongTensor *col, THDoubleTensor *weight);
