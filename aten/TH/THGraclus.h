void       THTensor_graclus(THLongTensor *self, THLongTensor *row, THLongTensor *col);

void   THByteTensor_graclus(THLongTensor *self, THLongTensor *row, THLongTensor *col,   THByteTensor *weight);
void   THCharTensor_graclus(THLongTensor *self, THLongTensor *row, THLongTensor *col,   THCharTensor *weight);
void  THShortTensor_graclus(THLongTensor *self, THLongTensor *row, THLongTensor *col,  THShortTensor *weight);
void    THIntTensor_graclus(THLongTensor *self, THLongTensor *row, THLongTensor *col,    THIntTensor *weight);
void   THLongTensor_graclus(THLongTensor *self, THLongTensor *row, THLongTensor *col,   THLongTensor *weight);
void  THFloatTensor_graclus(THLongTensor *self, THLongTensor *row, THLongTensor *col,  THFloatTensor *weight);
void THDoubleTensor_graclus(THLongTensor *self, THLongTensor *row, THLongTensor *col, THDoubleTensor *weight);
