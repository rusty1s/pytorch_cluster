void cluster_serial(THLongTensor *output, THLongTensor *row, THLongTensor *col, THLongTensor *degree);

void cluster_serial_Float (THLongTensor *output, THLongTensor *row, THLongTensor *col, THLongTensor *degree,  THFloatTensor *weight);
void cluster_serial_Double(THLongTensor *output, THLongTensor *row, THLongTensor *col, THLongTensor *degree, THDoubleTensor *weight);
void cluster_serial_Byte  (THLongTensor *output, THLongTensor *row, THLongTensor *col, THLongTensor *degree,   THByteTensor *weight);
void cluster_serial_Char  (THLongTensor *output, THLongTensor *row, THLongTensor *col, THLongTensor *degree,   THCharTensor *weight);
void cluster_serial_Short (THLongTensor *output, THLongTensor *row, THLongTensor *col, THLongTensor *degree,  THShortTensor *weight);
void cluster_serial_Int   (THLongTensor *output, THLongTensor *row, THLongTensor *col, THLongTensor *degree,    THIntTensor *weight);
void cluster_serial_Long  (THLongTensor *output, THLongTensor *row, THLongTensor *col, THLongTensor *degree,   THLongTensor *weight);
