#include <TH/TH.h>

#define grid_cluster TH_CONCAT_2(grid_cluster_, Real)

#include "generic/grid_cpu.c"
#include "THGenerateAllTypes.h"
