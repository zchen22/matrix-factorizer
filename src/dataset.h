#ifndef DATASET_H_
#define DATASET_H_

// C++ headers
#include <string>
#include <vector>

// Porject headers
#include "logger.h"
#include "record.h"
#include "user.h"

class Dataset {
 public:
  Dataset(const std::string name, const std::string filename, Logger* logger);
  ~Dataset();
  // Getters
  int GetUserId(const int record_id) const {
    return record_vector_[record_id].user_id;
  }
  int GetItemId(const int record_id) const {
    return record_vector_[record_id].item_id;
  }
  float GetRating(const int record_id) const {
    return record_vector_[record_id].rating;
  }
  int GetNumRecords() const { return record_vector_.size(); }
  int GetNumUsers() const { return num_users_; }
  int GetNumItems() const { return num_items_; }
  const int* GetUserIdDev() const { return user_id_vector_dev_; }
  const int* GetItemIdDev() const { return item_id_vector_dev_; }
  const float* GetRatingDev() const { return rating_vector_dev_; }
  const int* GetUserRecordBaseDev() const {
    return user_record_base_vector_dev_;
  }
  const int* GetUserNumRecordsDev() const {
    return user_num_records_vector_dev_;
  }
  // Load records
  int Load();
  int LoadFromText();
  int LoadFromBinary();
  // Shuffle
  int Shuffle();
  int ShuffleUsers();
  // Collect user information
  int CollectUserInfo();
  // Generate COO vectors
  int GenerateCoo();
  int GenerateCooByUsers();
  // GPU memory
  int AllocateCooGpu();
  int AllocateUserInfoGpu();
  int CopyCooToGpu();
  int CopyUserInfoToGpu();
 private:
  // Name
  std::string name_;
  // Data
  std::vector<Record> record_vector_;
  int num_users_;
  int num_items_;
  // COO vectors
  std::vector<int> user_id_vector_;
  std::vector<int> item_id_vector_;
  std::vector<float> rating_vector_;
  // User information
  std::vector<User*> user_vector_;
  std::vector<int> user_record_base_vector_;
  std::vector<int> user_num_records_vector_;
  // GPU memory objects
  int* user_id_vector_dev_;
  int* item_id_vector_dev_;
  float* rating_vector_dev_;
  int* user_record_base_vector_dev_;
  int* user_num_records_vector_dev_;
  // File that stores the data
  std::string filename_;
  // Logger
  Logger* logger_;
};

#endif

