void cluster_grid_Float (int C, THLongTensor *output, THFloatTensor  *position, THFloatTensor  *size, THLongTensor *count);
void cluster_grid_Double(int C, THLongTensor *output, THDoubleTensor *position, THDoubleTensor *size, THLongTensor *count);
void cluster_grid_Byte  (int C, THLongTensor *output, THByteTensor   *position, THByteTensor   *size, THLongTensor *count);
void cluster_grid_Char  (int C, THLongTensor *output, THCharTensor   *position, THCharTensor   *size, THLongTensor *count);
void cluster_grid_Short (int C, THLongTensor *output, THShortTensor  *position, THShortTensor  *size, THLongTensor *count);
void cluster_grid_Int   (int C, THLongTensor *output, THIntTensor    *position, THIntTensor    *size, THLongTensor *count);
void cluster_grid_Long  (int C, THLongTensor *output, THLongTensor   *position, THLongTensor   *size, THLongTensor *count);

void cluster_random(THLongTensor *output, THLongTensor *row, THLongTensor *col, THLongTensor *degree);
