void serial_cluster(THLongTensor *output, THLongTensor *row, THLongTensor *col, THLongTensor *degree);

void serial_cluster_Float (THLongTensor *output, THLongTensor *row, THLongTensor *col,  THFloatTensor *weight, THLongTensor *degree);
void serial_cluster_Double(THLongTensor *output, THLongTensor *row, THLongTensor *col, THDoubleTensor *weight, THLongTensor *degree);
void serial_cluster_Byte  (THLongTensor *output, THLongTensor *row, THLongTensor *col,   THByteTensor *weight, THLongTensor *degree);
void serial_cluster_Char  (THLongTensor *output, THLongTensor *row, THLongTensor *col,   THCharTensor *weight, THLongTensor *degree);
void serial_cluster_Short (THLongTensor *output, THLongTensor *row, THLongTensor *col,  THShortTensor *weight, THLongTensor *degree);
void serial_cluster_Int   (THLongTensor *output, THLongTensor *row, THLongTensor *col,    THIntTensor *weight, THLongTensor *degree);
void serial_cluster_Long  (THLongTensor *output, THLongTensor *row, THLongTensor *col,   THLongTensor *weight, THLongTensor *degree);
