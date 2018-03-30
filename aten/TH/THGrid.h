void THByteGrid(THLongTensor *cluster,
                THByteTensor *pos,
                THByteTensor *size,
                THLongTensor *count);

void THCharGrid(THLongTensor *cluster,
                THCharTensor *pos,
                THCharTensor *size,
                THLongTensor *count);

void THShortGrid(THLongTensor *cluster,
                 THShortTensor *pos,
                 THShortTensor *size,
                 THLongTensor *count);

void THIntGrid(THLongTensor *cluster,
               THIntTensor *pos,
               THIntTensor *size,
               THLongTensor *count);

void THLongGrid(THLongTensor *cluster,
                THLongTensor *pos,
                THLongTensor *size,
                THLongTensor *count);

void THFloatGrid(THLongTensor *cluster,
                 THFloatTensor *pos,
                 THFloatTensor *size,
                 THLongTensor *count);

void THDoubleGrid(THLongTensor *cluster,
                  THDoubleTensor *pos,
                  THDoubleTensor *size,
                  THLongTensor *count);
