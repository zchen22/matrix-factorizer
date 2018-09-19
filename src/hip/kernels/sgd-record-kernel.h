#ifndef SGD_RECORD_KERNEL_H_
#define SGD_RECORD_KERNEL_H_

#include "hip/hip_runtime.h"

__global__ void SgdRecord(const int* __restrict__ user_ids,
                          const int* __restrict__ item_ids,
                          const float* __restrict__ ratings,
                          const int num_records,
                          const int num_features,
                          const float learning_rate,
                          const float regularization_factor,
                          float* __restrict__ user_features,
                          float* __restrict__ item_features);

#endif

