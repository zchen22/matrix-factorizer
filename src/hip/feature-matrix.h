#ifndef FEATURE_MATRIX_H_
#define FEATURE_MATRIX_H_

// C++ headers
#include <cstdint>
#include <string>
#include <vector>

// Project headers
#include "configuration-set.h"
#include "logger.h"

class FeatureMatrix {
 public:
  FeatureMatrix(const std::string name, const int num_rows, ConfigurationSet* config, Logger* logger);
  ~FeatureMatrix();
  // Getters
  const std::vector<float>& GetFeatureVector() const { return feature_vector_; }
  const std::vector<float>& GetGradientVector() const { return gradient_vector_; }
  const std::vector<int16_t>& GetFeatureHalfVector() const { return feature_half_vector_; }
  const std::vector<int16_t>& GetGradientHalfVector() const { return gradient_half_vector_; }
  float* GetFeatureDev() const { return feature_vector_dev_; }
  float* GetGradientDev() const { return gradient_vector_dev_; }
  int16_t* GetFeatureHalfDev() const { return feature_half_vector_dev_; }
  int16_t* GetGradientHalfDev() const { return gradient_half_vector_dev_; }
  uint8_t* GetLockDev() const { return lock_vector_dev_; }
  // Setters
  void SetFeature(const int row, const int col, const float v) { feature_vector_[row * config_->num_features + col] = v; }
  void SetGradient(const int row, const int col, const float v) { gradient_vector_[row * config_->num_features + col] = v; }
  // Initialize features
  int Initialize();
  // CPU/GPU memory transfer
  int AllocateGpuMemory();
  int CopyToGpu();
  int CopyToCpu();
  // Dump to file
  int DumpToFile();
 private:
  // Name
  std::string name_;
  // 1D data
  std::vector<float> feature_vector_;
  std::vector<float> gradient_vector_;
  std::vector<int16_t> feature_half_vector_;
  std::vector<int16_t> gradient_half_vector_;
  std::vector<uint8_t> lock_vector_;
  // Metadata
  int num_rows_;
  ConfigurationSet* config_;
  // GPU memory objects
  float* feature_vector_dev_;
  float* gradient_vector_dev_;
  int16_t* feature_half_vector_dev_;
  int16_t* gradient_half_vector_dev_;
  uint8_t* lock_vector_dev_;
  // File that stores features
  std::string filename_;
  // Logger
  Logger* logger_;
};

#endif

