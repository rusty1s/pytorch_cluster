#include <TH/TH.h>

#define THGrid_(NAME) TH_CONCAT_4(TH,Real,Grid_,NAME)
#define DATA(TENSOR) TENSOR->storage->data + TENSOR->storageOffset

#include "generic/THGrid.c"
#include "THGenerateAllTypes.h"
