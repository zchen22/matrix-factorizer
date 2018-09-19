#ifndef SGD_ROW_KERNEL_H_
#define SGD_ROW_KERNEL_H_

#include "hip/hip_runtime.h"

__global__ void SgdRow(const int* __restrict__ user_ids,
                       const int* __restrict__ item_ids,
                       const float* __restrict__ ratings,
                       const int num_users,
                       const int* __restrict__ user_record_base,
                       const int* __restrict__ user_num_records,
                       const int num_features,
                       const float learning_rate,
                       const float regularization_factor,
                       float* __restrict__ user_features,
                       float* __restrict__ item_features);

#endif

