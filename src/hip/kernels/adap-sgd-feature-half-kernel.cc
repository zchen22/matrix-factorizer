#include "adap-sgd-feature-half-kernel.h"

__global__ void AdapSgdFeatureHalf(const int* __restrict__ user_ids,
                                   const int* __restrict__ item_ids,
                                   const __half* __restrict__ ratings,
                                   const int num_features,
                                   const __half learning_rate,
                                   const __half regularization_factor,
                                   __half* __restrict__ user_features,
                                   __half* __restrict__ item_features,
                                   __half* __restrict__ user_grads,
                                   __half* __restrict__ item_grads) {
  // Compute error
  const int user_id = user_ids[blockIdx.x];
  const int item_id = item_ids[blockIdx.x];
  const int user_feature_idx = user_id * num_features + threadIdx.x;
  const int item_feature_idx = item_id * num_features + threadIdx.x;
  const __half user_feature = user_features[user_feature_idx];
  const __half item_feature = item_features[item_feature_idx];
  HIP_DYNAMIC_SHARED(__half, partial_sums)
  partial_sums[threadIdx.x] = __hmul(user_feature, item_feature);
  __syncthreads();
  if (threadIdx.x < warpSize) {
    for (int i = threadIdx.x + warpSize; i < blockDim.x; i += warpSize) {
      partial_sums[threadIdx.x] = __hadd(partial_sums[threadIdx.x], partial_sums[i]);
    }
  }
  __syncthreads();
  float predicted_rating = __half2float(partial_sums[threadIdx.x]);
  if (threadIdx.x < warpSize) {
    #pragma unroll
    for (int i = warpSize / 2; i >= 1; i /= 2) {
      predicted_rating += __shfl_xor(predicted_rating, i, warpSize);
    }
  }
  __syncthreads();
  __shared__ __half error;
  if (threadIdx.x == 0) {
    error = __hsub(__float2half(predicted_rating), ratings[blockIdx.x]);
  }
  __syncthreads();
  // Update features
  const __half user_grad = __hadd(__hmul(regularization_factor, user_feature), __hmul(error, item_feature));
  const __half item_grad = __hadd(__hmul(regularization_factor, item_feature), __hmul(error, user_feature));
  __half new_user_grad = __hadd(user_grads[user_feature_idx], __hmul(user_grad, user_grad));
  __half non_zero_new_user_grad = new_user_grad;
  if ((__half_as_ushort(non_zero_new_user_grad) & 0x7fff) == 0) {
    non_zero_new_user_grad = __ushort_as_half(__half_as_ushort(non_zero_new_user_grad) | 0x0001);
  }
  __half new_item_grad = __hadd(item_grads[item_feature_idx], __hmul(item_grad, item_grad));
  __half non_zero_new_item_grad = new_item_grad;
  if ((__half_as_ushort(non_zero_new_item_grad) & 0x7fff) == 0) {
    non_zero_new_item_grad = __ushort_as_half(__half_as_ushort(non_zero_new_item_grad) | 0x0001);
  }
  const __half new_user_learning_rate = hdiv(learning_rate, hsqrt(non_zero_new_user_grad));
  const __half new_item_learning_rate = hdiv(learning_rate, hsqrt(non_zero_new_item_grad));
  const __half new_user_feature = __hsub(user_feature, __hmul(new_user_learning_rate, user_grad));
  const __half new_item_feature = __hsub(item_feature, __hmul(new_item_learning_rate, item_grad));
  user_features[user_feature_idx] = new_user_feature;
  item_features[item_feature_idx] = new_item_feature;
  user_grads[user_feature_idx] = new_user_grad;
  item_grads[item_feature_idx] = new_item_grad;
}

