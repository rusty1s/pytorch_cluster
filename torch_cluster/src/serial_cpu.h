void cluster_serial(THLongTensor *output, THLongTensor *row, THLongTensor *col, THLongTensor *degree);

void cluster_serial_Float (THLongTensor *output, THLongTensor *row, THLongTensor *col,  THFloatTensor *weight, THLongTensor *degree);
void cluster_serial_Double(THLongTensor *output, THLongTensor *row, THLongTensor *col, THDoubleTensor *weight, THLongTensor *degree);
void cluster_serial_Byte  (THLongTensor *output, THLongTensor *row, THLongTensor *col,   THByteTensor *weight, THLongTensor *degree);
void cluster_serial_Char  (THLongTensor *output, THLongTensor *row, THLongTensor *col,   THCharTensor *weight, THLongTensor *degree);
void cluster_serial_Short (THLongTensor *output, THLongTensor *row, THLongTensor *col,  THShortTensor *weight, THLongTensor *degree);
void cluster_serial_Int   (THLongTensor *output, THLongTensor *row, THLongTensor *col,    THIntTensor *weight, THLongTensor *degree);
void cluster_serial_Long  (THLongTensor *output, THLongTensor *row, THLongTensor *col,   THLongTensor *weight, THLongTensor *degree);
