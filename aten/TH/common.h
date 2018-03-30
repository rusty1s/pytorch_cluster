#ifndef TH_COMMON_INC
#define TH_COMMON_INC

#define THTensor_getData(TENSOR) TENSOR->storage->data + TENSOR->storageOffset

#endif  // TH_COMMON_INC
