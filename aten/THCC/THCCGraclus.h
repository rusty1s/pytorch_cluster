void       THCCTensor_graclus(THCudaLongTensor *self, THCudaLongTensor *row, THCudaLongTensor *col);

void   THCCByteTensor_graclus(THCudaLongTensor *self, THCudaLongTensor *row, THCudaLongTensor *col,   THCudaByteTensor *weight);
void   THCCCharTensor_graclus(THCudaLongTensor *self, THCudaLongTensor *row, THCudaLongTensor *col,   THCudaCharTensor *weight);
void  THCCShortTensor_graclus(THCudaLongTensor *self, THCudaLongTensor *row, THCudaLongTensor *col,  THCudaShortTensor *weight);
void    THCCIntTensor_graclus(THCudaLongTensor *self, THCudaLongTensor *row, THCudaLongTensor *col,    THCudaIntTensor *weight);
void   THCCLongTensor_graclus(THCudaLongTensor *self, THCudaLongTensor *row, THCudaLongTensor *col,   THCudaLongTensor *weight);
void  THCCFloatTensor_graclus(THCudaLongTensor *self, THCudaLongTensor *row, THCudaLongTensor *col,       THCudaTensor *weight);
void THCCDoubleTensor_graclus(THCudaLongTensor *self, THCudaLongTensor *row, THCudaLongTensor *col, THCudaDoubleTensor *weight);
