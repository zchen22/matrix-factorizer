#ifndef SGD_FEATURE_HALF_KERNEL_H_
#define SGD_FEATURE_HALF_KERNEL_H_

#include "hip/hip_fp16.h"
#include "hip/hip_runtime.h"

__global__ void SgdFeatureHalf(const int* __restrict__ user_ids,
                               const int* __restrict__ item_ids,
                               const __half* __restrict__ ratings,
                               const int num_features,
                               const __half learning_rate,
                               const __half regularization_factor,
                               __half* __restrict__ user_features,
                               __half* __restrict__ item_features);

#endif

