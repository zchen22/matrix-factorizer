#include "sgd-row-kernel.h"

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
                       float* __restrict__ item_features) {
  const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (thread_id >= num_users) {
    return;
  }
  const int record_base = user_record_base[thread_id];
  const int num_records = user_num_records[thread_id];
  for (int record_id = record_base; record_id < record_base + num_records;
       ++record_id) {
    const int64_t user_id = user_ids[record_id];
    const int64_t item_id = item_ids[record_id];
    float error = 0;
    for (int feature_id = 0; feature_id < num_features; ++feature_id) {
      error += user_features[user_id * num_features + feature_id] *
          item_features[item_id * num_features + feature_id];
    }
    error -= ratings[record_id];
    for (int feature_id = 0; feature_id < num_features; ++feature_id) {
      const float user_feature =
          user_features[user_id * num_features + feature_id];
      const float item_feature =
          item_features[item_id * num_features + feature_id];
      const float new_user_feature = user_feature - learning_rate *
          (regularization_factor * user_feature + error * item_feature);
      const float new_item_feature = item_feature - learning_rate *
          (regularization_factor * item_feature + error * user_feature);
      user_features[user_id * num_features + feature_id] = new_user_feature;
      item_features[item_id * num_features + feature_id] = new_item_feature;
    }
  }
}

