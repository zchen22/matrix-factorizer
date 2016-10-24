#ifndef ADAP_SGD_FEATURE_KERNEL_H_
#define ADAP_SGD_FEATURE_KERNEL_H_

__global__ void AdapSgdFeature(const int* __restrict__ user_ids,
                               const int* __restrict__ item_ids,
                               const float* __restrict__ ratings,
                               const int num_features,
                               const float learning_rate,
                               const float regularization_factor,
                               float* __restrict__ user_features,
                               float* __restrict__ item_features,
                               float* __restrict__ user_grads,
                               float* __restrict__ item_grads);

#endif

