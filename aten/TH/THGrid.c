#include <TH/TH.h>

#define THGrid_ TH_CONCAT_3(TH,Real,Grid)
#define DATA(TENSOR) TENSOR->storage->data + TENSOR->storageOffset

#include "generic/THGrid.c"
#include "THGenerateAllTypes.h"
