#include "sgd-record-lock-kernel.h"

__global__ void SgdRecordLock(const int* __restrict__ user_ids,
                              const int* __restrict__ item_ids,
                              const float* __restrict__ ratings,
                              const int num_records,
                              const int num_features,
                              const float learning_rate,
                              const float regularization_factor,
                              int* user_busy_bits,
                              int* item_busy_bits,
                              float* __restrict__ user_features,
                              float* __restrict__ item_features) {
  const int64_t record_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (record_id >= num_records) {
    return;
  }
  const int user_id = user_ids[record_id];
  const int item_id = item_ids[record_id];
  bool done = 0;
  while (!done) {
    if (!atomicCAS(user_busy_bits + user_id, 0, 1) && !atomicCAS(item_busy_bits + item_id, 0, 1)) {
      // Compute error
      float error = -ratings[record_id];
      for (int feature_id = 0; feature_id < num_features; ++feature_id) {
        error += user_features[user_id * num_features + feature_id] * item_features[item_id * num_features + feature_id];
      }
      // Update features
      for (int feature_id = 0; feature_id < num_features; ++feature_id) {
        const float user_feature = user_features[user_id * num_features + feature_id];
        const float item_feature = item_features[item_id * num_features + feature_id];
        const float user_grad = regularization_factor * user_feature + error * item_feature;
        const float item_grad = regularization_factor * item_feature + error * user_feature;
        const float new_user_feature = user_feature - learning_rate * user_grad;
        const float new_item_feature = item_feature - learning_rate * item_grad;
        user_features[user_id * num_features + feature_id] = new_user_feature;
        item_features[item_id * num_features + feature_id] = new_item_feature;
      }
      done = ~done;
    }
    atomicExch(item_busy_bits + item_id, 0);
    atomicExch(user_busy_bits + user_id, 0);
  }
}

