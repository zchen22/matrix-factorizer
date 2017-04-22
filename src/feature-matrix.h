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
  FeatureMatrix(const std::string name, const int num_rows,
                ConfigurationSet* config, Logger* logger);
  ~FeatureMatrix();
  // Getters
  const std::vector<float>& GetFeatureVector(const int row) const {
    return feature_vector_[row];
  }
  float GetFeature(const int64_t index) const {
    return feature_1d_vector_[index];
  }
  const std::vector<float>& GetGradientVector(const int row) const {
    return gradient_vector_[row];
  }
  const std::vector<unsigned short>& GetFeatureHalf1dVector() const {
    return feature_half_1d_vector_;
  }
  const std::vector<float>& GetFeature1dVector() const {
    return feature_1d_vector_;
  }
  const std::vector<unsigned int>& GetParameter32bit1dVector() const {
    return parameter_32bit_1d_vector_;
  }
  unsigned short* GetFeatureHalfDev() const {
    return feature_half_1d_vector_dev_;
  }
  float* GetFeatureDev() const { return feature_1d_vector_dev_; }
  float* GetGradientDev() const { return gradient_1d_vector_dev_; }
  unsigned int* GetParameterDev() const {
    return parameter_32bit_1d_vector_dev_;
  }
  int* GetLockDev() const { return lock_vector_dev_; }
  // Setters
  void SetFeatureVector(const int row, const std::vector<float>& v) {
    feature_vector_[row] = v;
  }
  void SetGradientVector(const int row, const std::vector<float>& v) {
    gradient_vector_[row] = v;
  }
  // Initialize features
  int Initialize();
  // Flatten
  int Flatten();
  // CPU/GPU memory transfer
  int AllocateGpuMemory();
  int CopyToGpu();
  int CopyToCpuFlatten();
  // Dump to file
  int DumpToFile();
  int Dump1dToFile();
 private:
  // Name
  std::string name_;
  // Data
  std::vector<std::vector<float>> feature_vector_;
  std::vector<std::vector<float>> gradient_vector_;
  std::vector<int> lock_vector_;
  // 1D data
  std::vector<unsigned short> feature_half_1d_vector_;
  std::vector<float> feature_1d_vector_;
  std::vector<float> gradient_1d_vector_;
  std::vector<unsigned int> parameter_32bit_1d_vector_;
  std::vector<uint16_t> parameter_16bit_1d_vector_;
  // Metadata
  int num_rows_;
  ConfigurationSet* config_;
  // GPU memory objects
  unsigned short* feature_half_1d_vector_dev_;
  float* feature_1d_vector_dev_;
  float* gradient_1d_vector_dev_;
  unsigned int* parameter_32bit_1d_vector_dev_;
  uint16_t* parameter_16bit_1d_vector_dev_;
  int* lock_vector_dev_;
  // File that stores features
  std::string filename_;
  FILE* file_;
  // Logger
  Logger* logger_;
};

#endif

