#include "adap-sgd-feature-half-kernel.h"

__global__ void AdapSgdFeatureHalf(const int* __restrict__ user_ids,
                                   const int* __restrict__ item_ids,
                                   const float* __restrict__ ratings,
                                   const int num_features,
                                   const float learning_rate,
                                   const float regularization_factor,
                                   unsigned int* __restrict__ user_parameters,
                                   unsigned int* __restrict__ item_parameters) {
  // Compute error
  const int64_t user_id = user_ids[blockIdx.x];
  const int64_t item_id = item_ids[blockIdx.x];
  const int64_t user_parameter_idx = user_id * num_features + threadIdx.x;
  const int64_t item_parameter_idx = item_id * num_features + threadIdx.x;
  const unsigned int user_parameter = user_parameters[user_parameter_idx];
  const unsigned int item_parameter = item_parameters[item_parameter_idx];
  const float user_feature = __half2float(user_parameter & 0xffff);
  const float item_feature = __half2float(item_parameter & 0xffff);
  extern __shared__ float partial_sums[];
  partial_sums[threadIdx.x] = user_feature * item_feature;
  __syncthreads();
  if (threadIdx.x < warpSize) {
    for (int i = threadIdx.x + warpSize; i < blockDim.x; i += warpSize) {
      partial_sums[threadIdx.x] += partial_sums[i];
    }
  }
  __syncthreads();
  float predicted_rating = partial_sums[threadIdx.x];
  if (threadIdx.x < warpSize) {
    for (int i = warpSize / 2; i >= 1; i /= 2) {
      predicted_rating += __shfl_xor(predicted_rating, i, warpSize);
    }
  }
  __syncthreads();
  __shared__ float error;
  if (threadIdx.x == 0) {
    error = predicted_rating - ratings[blockIdx.x];
  }
  __syncthreads();
  // Update features
  const float user_grad = regularization_factor * user_feature +
      error * item_feature;
  const float item_grad = regularization_factor * item_feature +
      error * user_feature;
  const float new_user_grad = __half2float((user_parameter >> 16) & 0xffff) +
      user_grad * user_grad;
  const float new_item_grad = __half2float((item_parameter >> 16) & 0xffff) +
      item_grad * item_grad;
  const float new_user_learning_rate = learning_rate / sqrt(new_user_grad);
  const float new_item_learning_rate = learning_rate / sqrt(new_item_grad);
  const float new_user_feature = user_feature -
      new_user_learning_rate * user_grad;
  const float new_item_feature = item_feature -
      new_item_learning_rate * item_grad;
  const unsigned int new_user_parameter =
      (((unsigned int)__float2half_rn(new_user_grad)) << 16) |
      __float2half_rn(new_user_feature);
  const unsigned int new_item_parameter =
      (((unsigned int)__float2half_rn(new_item_grad)) << 16) |
      __float2half_rn(new_item_feature);
  user_parameters[user_parameter_idx] = new_user_parameter;
  item_parameters[item_parameter_idx] = new_item_parameter;
}

