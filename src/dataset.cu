#include "dataset.h"

// C++ headers
#include <algorithm>
#include <cassert>
#include <fstream>
#include <sstream>

Dataset::Dataset(const std::string name, const std::string filename,
                 Logger* logger)
    : num_users_(0), num_items_(0), user_id_vector_dev_(NULL),
      item_id_vector_dev_(NULL), rating_vector_dev_(NULL), logger_(logger) {
  name_ = name;
  filename_ = filename;
}

Dataset::~Dataset() {
}

int Dataset::Load() {
  if (filename_.compare(filename_.size() - 4, 4, ".txt") == 0) {
    LoadFromText();
  } else if (filename_.compare(filename_.size() - 4, 4, ".bin") == 0) {
    LoadFromBinary();
  } else {
    logger_->Error(stderr, "Unrecognized dataset file type\n");
  }
  return 0;
}

int Dataset::LoadFromText() {
  // Get the metadata
  std::ifstream file(filename_);
  if (!file.is_open()) {
    logger_->Error(stderr, "File '%s' cannot be opened!\n", filename_.c_str());
  }
  std::string line;
  int num_records = 0;
  while (std::getline(file, line)) {
    if (line[0] == '%') {
      continue;
    }
    std::istringstream line_stream(line);
    if (num_users_ == 0) {
      assert(line_stream >> num_users_);
    } else if (num_users_ > 0) {
      int val = 0;
      assert(line_stream >> val);
      assert(val == num_users_);
    }
    if (num_items_ == 0) {
      assert(line_stream >> num_items_);
    } else if (num_items_ > 0) {
      int val = 0;
      assert(line_stream >> val);
      assert(val == num_items_);
    }
    assert(line_stream >> num_records);
    record_vector_.reserve(num_records);
    break;
  }
  // Get the data in the COO format
  for (int i = 0; i < num_records; ++i) {
    assert(std::getline(file, line));
    int user_id = 0;
    int item_id = 0;
    float rating = 0;
    std::istringstream line_stream(line);
    assert(line_stream >> user_id);
    assert(line_stream >> item_id);
    assert(line_stream >> rating);
    record_vector_.push_back(Record(--user_id, --item_id, rating));
  }
  file.close();
  return 0;
}

int Dataset::LoadFromBinary() {
  // Get the metadata
  std::ifstream file(filename_, std::ifstream::binary);
  if (!file.is_open()) {
    logger_->Error(stderr, "File '%s' cannot be opened!\n", filename_.c_str());
  }
  union {
    char b[8];
    unsigned int u32;
    uint64_t u64;
    float f32;
    double f64;
  } bytes = { .u64 = 0 };
  file.read(bytes.b, 4);
  num_users_ = bytes.u32;
  bytes.u64 = 0;
  file.read(bytes.b, 4);
  num_items_ = bytes.u32;
  bytes.u64 = 0;
  file.read(bytes.b, 8);
  assert(bytes.u64 <= 0xffffffffLL);
  const int num_records = bytes.u64;
  record_vector_.reserve(num_records);
  // Get the data in the COO format
  for (int i = 0; i < num_records; ++i) {
    bytes.u64 = 0;
    file.read(bytes.b, 4);
    const int user_id = bytes.u32 - 1;
    bytes.u64 = 0;
    file.read(bytes.b, 4);
    const int item_id = bytes.u32 - 1;
    bytes.u64 = 0;
    file.read(bytes.b, 4);
    const float rating = bytes.f32;
    record_vector_.push_back(Record(user_id, item_id, rating));
  }
  file.close();
  return 0;
}

int Dataset::Shuffle() {
  logger_->Debug(stderr, "Shuffling records...\n");
  std::random_shuffle(record_vector_.begin(), record_vector_.end());
  logger_->Debug(stderr, "Records shuffled\n");
  return 0;
}

int Dataset::ShuffleUsers() {
  logger_->Debug(stderr, "Shuffling users...\n");
  std::random_shuffle(user_vector_.begin(), user_vector_.end());
  logger_->Debug(stderr, "Users shuffled\n");
  return 0;
}

int Dataset::CollectUserInfo() {
  user_vector_.assign(num_users_, NULL);
  for (int user_id = 0; user_id < num_users_; ++user_id) {
    user_vector_[user_id] = new User(user_id);
  }
  for (const auto& r : record_vector_) {
    const int user_id = r.user_id;
    user_vector_[user_id]->AddItemRatingPair(r.item_id, r.rating);
  }
  return 0;
}

int Dataset::GenerateCoo() {
  user_id_vector_.reserve(record_vector_.size());
  item_id_vector_.reserve(record_vector_.size());
  rating_vector_.reserve(record_vector_.size());
  for (const auto& r : record_vector_) {
    user_id_vector_.push_back(r.user_id);
    item_id_vector_.push_back(r.item_id);
    rating_vector_.push_back(r.rating);
  }
  return 0;
}

int Dataset::GenerateCooByUsers() {
  user_id_vector_.reserve(record_vector_.size());
  item_id_vector_.reserve(record_vector_.size());
  rating_vector_.reserve(record_vector_.size());
  for (const auto user : user_vector_) {
    user_record_base_vector_.push_back(user_id_vector_.size());
    for (const auto& item_rating_pair: user->GetItemRatingMap()) {
      user_id_vector_.push_back(user->GetId());
      item_id_vector_.push_back(item_rating_pair.first);
      rating_vector_.push_back(item_rating_pair.second);
    }
    user_num_records_vector_.push_back(
        user_id_vector_.size() - user_record_base_vector_.back());
  }
  return 0;
}

int Dataset::AllocateCooGpu() {
  cudaError_t e = cudaSuccess;
  e = cudaMalloc(&user_id_vector_dev_,
                 user_id_vector_.size() * sizeof user_id_vector_[0]);
  logger_->CheckCudaError(e);
  e = cudaMalloc(&item_id_vector_dev_,
                 item_id_vector_.size() * sizeof item_id_vector_[0]);
  logger_->CheckCudaError(e);
  e = cudaMalloc(&rating_vector_dev_,
                 rating_vector_.size() * sizeof rating_vector_[0]);
  logger_->CheckCudaError(e);
  return 0;
}

int Dataset::AllocateUserInfoGpu() {
  cudaError_t e = cudaSuccess;
  e = cudaMalloc(&user_record_base_vector_dev_,
                 user_record_base_vector_.size() *
                     sizeof user_record_base_vector_[0]);
  logger_->CheckCudaError(e);
  e = cudaMalloc(&user_num_records_vector_dev_,
                 user_num_records_vector_.size() *
                     sizeof user_num_records_vector_[0]);
  logger_->CheckCudaError(e);
  return 0;
}

int Dataset::CopyCooToGpu() {
  cudaError_t e = cudaSuccess;
  e = cudaMemcpy(user_id_vector_dev_, user_id_vector_.data(),
                 user_id_vector_.size() * sizeof user_id_vector_[0],
                 cudaMemcpyHostToDevice);
  logger_->CheckCudaError(e);
  e = cudaMemcpy(item_id_vector_dev_, item_id_vector_.data(),
                 item_id_vector_.size() * sizeof item_id_vector_[0],
                 cudaMemcpyHostToDevice);
  logger_->CheckCudaError(e);
  e = cudaMemcpy(rating_vector_dev_, rating_vector_.data(),
                 rating_vector_.size() * sizeof rating_vector_[0],
                 cudaMemcpyHostToDevice);
  logger_->CheckCudaError(e);
  return 0;
}

int Dataset::CopyUserInfoToGpu() {
  cudaError_t e = cudaSuccess;
  e = cudaMemcpy(user_record_base_vector_dev_, user_record_base_vector_.data(),
                 user_record_base_vector_.size() *
                     sizeof user_record_base_vector_[0],
                 cudaMemcpyHostToDevice);
  logger_->CheckCudaError(e);
  e = cudaMemcpy(user_num_records_vector_dev_, user_num_records_vector_.data(),
                 user_num_records_vector_.size() *
                     sizeof user_num_records_vector_[0],
                 cudaMemcpyHostToDevice);
  logger_->CheckCudaError(e);
  return 0;
}

