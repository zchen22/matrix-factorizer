#include "feature-matrix.h"

// C++ headers
#include <ctime>
#include <random>

// Project headers
#include "common.h"

FeatureMatrix::FeatureMatrix(const std::string name, const int num_rows,
                             ConfigurationSet* config, Logger* logger)
    : num_rows_(num_rows), config_(config), feature_1d_vector_dev_(NULL),
      file_(NULL), logger_(logger) {
  name_ = name;
  // Allocate memory for feature vector
  feature_vector_.assign(num_rows_, std::vector<float>());
  gradient_vector_.assign(num_rows_, std::vector<float>());
  for (int i = 0; i < num_rows_; ++i) {
    feature_vector_[i].assign(config_->num_features, 0);
    gradient_vector_[i].assign(config_->num_features, 0);
  }
  lock_vector_.assign(num_rows_, 0);
  // Generate filename
  time_t timer = time(NULL);
  tm local_time = *localtime(&timer);
  char time_string[1024] = {0};
  snprintf(time_string, sizeof time_string, "%02d-%02d-%02d-%02d-%02d",
           local_time.tm_mon, local_time.tm_mday, local_time.tm_hour,
           local_time.tm_min, local_time.tm_sec);
  filename_ = name_ + "-" + time_string + ".txt";
}

FeatureMatrix::~FeatureMatrix() {
}

int FeatureMatrix::Initialize() {
  std::default_random_engine rng;
  std::uniform_real_distribution<float> dist(0, 0.1);
  for (int i = 0; i < num_rows_; ++i) {
    for (int j = 0; j < config_->num_features; ++j) {
      feature_vector_[i][j] = dist(rng);
      gradient_vector_[i][j] = 0;
    }
  }
  return 0;
}

int FeatureMatrix::Flatten() {
  if (config_->precision == ConfigurationSet::kPrecisionHalf) {
    if (config_->gd_mode == ConfigurationSet::kGdModeMiniBatchSgd) {
      feature_half_1d_vector_.reserve(num_rows_ * config_->num_features);
      for (int i = 0; i < num_rows_; ++i) {
        for (int j = 0; j < config_->num_features; ++j) {
          feature_half_1d_vector_.push_back(Float2Half(feature_vector_[i][j]));
        }
      }
    } else if (config_->gd_mode == ConfigurationSet::kGdModeAdapSgd) {
      parameter_32bit_1d_vector_.reserve(num_rows_ * config_->num_features);
      for (int i = 0; i < num_rows_; ++i) {
        for (int j = 0; j < config_->num_features; ++j) {
          const unsigned int feature = Float2Half(feature_vector_[i][j]);
          const unsigned int gradient = Float2Half(gradient_vector_[i][j]);
          parameter_32bit_1d_vector_.push_back((gradient << 16) | feature);
        }
      }
    }
  } else if (config_->precision == ConfigurationSet::kPrecisionSingle) {
    feature_1d_vector_.reserve(num_rows_ * config_->num_features);
    gradient_1d_vector_.reserve(num_rows_ * config_->num_features);
    for (int i = 0; i < num_rows_; ++i) {
      for (int j = 0; j < config_->num_features; ++j) {
        feature_1d_vector_.push_back(feature_vector_[i][j]);
        gradient_1d_vector_.push_back(gradient_vector_[i][j]);
      }
    }
  }
  return 0;
}

int FeatureMatrix::AllocateGpuMemory() {
  cudaError_t e = cudaSuccess;
  if (config_->precision == ConfigurationSet::kPrecisionMini) {
    assert(config_->gd_mode == ConfigurationSet::kGdModeAdapSgd);
    e = cudaMalloc(&parameter_16bit_1d_vector_dev_,
                   parameter_16bit_1d_vector_.size() *
                       sizeof parameter_16bit_1d_vector_[0]);
    logger_->CheckCudaError(e);
  } else if (config_->precision == ConfigurationSet::kPrecisionHalf) {
    if (config_->gd_mode == ConfigurationSet::kGdModeMiniBatchSgd) {
      e = cudaMalloc(&feature_half_1d_vector_dev_,
                     feature_half_1d_vector_.size() *
                         sizeof feature_half_1d_vector_[0]);
      logger_->CheckCudaError(e);
    } else if (config_->gd_mode == ConfigurationSet::kGdModeAdapSgd) {
      e = cudaMalloc(&parameter_32bit_1d_vector_dev_,
                     parameter_32bit_1d_vector_.size() *
                         sizeof parameter_32bit_1d_vector_[0]);
      logger_->CheckCudaError(e);
    }
  } else if (config_->precision == ConfigurationSet::kPrecisionSingle) {
    e = cudaMalloc(&feature_1d_vector_dev_,
                   feature_1d_vector_.size() * sizeof feature_1d_vector_[0]);
    logger_->CheckCudaError(e);
    e = cudaMalloc(&gradient_1d_vector_dev_,
                   gradient_1d_vector_.size() * sizeof gradient_1d_vector_[0]);
    logger_->CheckCudaError(e);
  }
  if (config_->lock) {
    e = cudaMalloc(&lock_vector_dev_,
                   lock_vector_.size() * sizeof lock_vector_[0]);
    logger_->CheckCudaError(e);
  }
  return 0;
}

int FeatureMatrix::CopyToGpu() {
  cudaError_t e = cudaSuccess;
  if (config_->precision == ConfigurationSet::kPrecisionMini) {
    assert(config_->gd_mode == ConfigurationSet::kGdModeAdapSgd);
    e = cudaMemcpy(parameter_16bit_1d_vector_dev_,
                   parameter_16bit_1d_vector_.data(),
                   parameter_16bit_1d_vector_.size() *
                       sizeof parameter_16bit_1d_vector_[0],
                   cudaMemcpyHostToDevice);
    logger_->CheckCudaError(e);
  } else if (config_->precision == ConfigurationSet::kPrecisionHalf) {
    if (config_->gd_mode == ConfigurationSet::kGdModeMiniBatchSgd) {
      e = cudaMemcpy(feature_half_1d_vector_dev_,
                     feature_half_1d_vector_.data(),
                     feature_half_1d_vector_.size() *
                         sizeof feature_half_1d_vector_[0],
                     cudaMemcpyHostToDevice);
      logger_->CheckCudaError(e);
    } else if (config_->gd_mode == ConfigurationSet::kGdModeAdapSgd) {
      e = cudaMemcpy(parameter_32bit_1d_vector_dev_,
                     parameter_32bit_1d_vector_.data(),
                     parameter_32bit_1d_vector_.size() *
                         sizeof parameter_32bit_1d_vector_[0],
                     cudaMemcpyHostToDevice);
      logger_->CheckCudaError(e);
    }
  } else if (config_->precision == ConfigurationSet::kPrecisionSingle) {
    e = cudaMemcpy(feature_1d_vector_dev_, feature_1d_vector_.data(),
                   feature_1d_vector_.size() * sizeof feature_1d_vector_[0],
                   cudaMemcpyHostToDevice);
    logger_->CheckCudaError(e);
    e = cudaMemcpy(gradient_1d_vector_dev_, gradient_1d_vector_.data(),
                   gradient_1d_vector_.size() * sizeof gradient_1d_vector_[0],
                   cudaMemcpyHostToDevice);
    logger_->CheckCudaError(e);
  }
  if (config_->lock) {
    e = cudaMemcpy(lock_vector_dev_, lock_vector_.data(),
                   lock_vector_.size() * sizeof lock_vector_[0],
                   cudaMemcpyHostToDevice);
    logger_->CheckCudaError(e);
  }
  return 0;
}

int FeatureMatrix::CopyToCpuFlatten() {
  cudaError_t e = cudaSuccess;
  if (config_->precision == ConfigurationSet::kPrecisionMini) {
    assert(config_->gd_mode == ConfigurationSet::kGdModeAdapSgd);
    e = cudaMemcpy(parameter_16bit_1d_vector_.data(),
                  parameter_16bit_1d_vector_dev_,
                  parameter_16bit_1d_vector_.size() *
                      sizeof parameter_16bit_1d_vector_[0],
                  cudaMemcpyDeviceToHost);
    logger_->CheckCudaError(e);
  } else if (config_->precision == ConfigurationSet::kPrecisionHalf) {
    if (config_->gd_mode == ConfigurationSet::kGdModeMiniBatchSgd) {
      e = cudaMemcpy(feature_half_1d_vector_.data(),
                     feature_half_1d_vector_dev_,
                     feature_half_1d_vector_.size() *
                         sizeof feature_half_1d_vector_[0],
                     cudaMemcpyDeviceToHost);
      logger_->CheckCudaError(e);
    } else if (config_->gd_mode == ConfigurationSet::kGdModeAdapSgd) {
      e = cudaMemcpy(parameter_32bit_1d_vector_.data(),
                     parameter_32bit_1d_vector_dev_,
                     parameter_32bit_1d_vector_.size() *
                         sizeof parameter_32bit_1d_vector_[0],
                     cudaMemcpyDeviceToHost);
      logger_->CheckCudaError(e);
    }
  } else if (config_->precision == ConfigurationSet::kPrecisionSingle) {
    e = cudaMemcpy(feature_1d_vector_.data(), feature_1d_vector_dev_,
                   feature_1d_vector_.size() * sizeof feature_1d_vector_[0],
                   cudaMemcpyDeviceToHost);
    logger_->CheckCudaError(e);
    e = cudaMemcpy(gradient_1d_vector_.data(), gradient_1d_vector_dev_,
                   gradient_1d_vector_.size() * sizeof gradient_1d_vector_[0],
                   cudaMemcpyDeviceToHost);
    logger_->CheckCudaError(e);
  }
  return 0;
}

int FeatureMatrix::DumpToFile() {
  logger_->Info(stderr, "Dumping features '%s'...\n", name_.c_str());
  FILE* f = fopen(filename_.c_str(), "w");
  for (int row = 0; row < num_rows_; ++row) {
    for (int feature_id = 0; feature_id < config_->num_features; ++feature_id) {
      fprintf(f, "%10.2g ", feature_vector_[row][feature_id]);
    }
    fprintf(f, "\n");
  }
  fclose(f);
  logger_->Info(stderr, "Features '%s' dumped to file '%s'\n",
                name_.c_str(), filename_.c_str());
  return 0;
}

int FeatureMatrix::Dump1dToFile() {
  logger_->Info(stderr, "Dumping feature matrix '%s'...\n", name_.c_str());
  FILE* f = fopen(filename_.c_str(), "w");
  for (int64_t row = 0; row < num_rows_; ++row) {
    for (int feature_id = 0; feature_id < config_->num_features; ++feature_id) {
      if (config_->precision == ConfigurationSet::kPrecisionMini) {
        assert(config_->gd_mode == ConfigurationSet::kGdModeAdapSgd);
        assert(0);
      } else if (config_->precision == ConfigurationSet::kPrecisionHalf) {
        if (config_->gd_mode == ConfigurationSet::kGdModeMiniBatchSgd) {
          const float feature = Half2Float(feature_half_1d_vector_
              [row * config_->num_features + feature_id]);
          fprintf(f, "%10.2g ", feature);
        } else if (config_->gd_mode == ConfigurationSet::kGdModeAdapSgd) {
          const unsigned int parameter = parameter_32bit_1d_vector_
              [row * config_->num_features + feature_id];
          fprintf(f, "%10.2g ", Half2Float(parameter & 0xffff));
        }
      } else if (config_->precision == ConfigurationSet::kPrecisionSingle) {
        const float feature = feature_1d_vector_
            [row * config_->num_features + feature_id];
        fprintf(f, "%10.2g ", feature);
      }
    }
    fprintf(f, "\n");
  }
  fclose(f);
  logger_->Info(stderr, "Feature matrix '%s' dumped to file '%s'\n",
                name_.c_str(), filename_.c_str());
  return 0;
}

