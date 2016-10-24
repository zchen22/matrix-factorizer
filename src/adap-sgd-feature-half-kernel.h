#ifndef ADAP_SGD_FEATURE_HALF_KERNEL_H_
#define ADAP_SGD_FEATURE_HALF_KERNEL_H_

__global__ void AdapSgdFeatureHalf(const int* __restrict__ user_ids,
                                   const int* __restrict__ item_ids,
                                   const float* __restrict__ ratings,
                                   const int num_features,
                                   const float learning_rate,
                                   const float regularization_factor,
                                   unsigned int* __restrict__ user_parameters,
                                   unsigned int* __restrict__ item_parameters);

#endif

