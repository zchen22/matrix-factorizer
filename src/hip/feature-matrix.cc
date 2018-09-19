#include "feature-matrix.h"

// C++ headers
#include <ctime>
#include <random>

// Project headers
#include "varied-precision-float.h"

FeatureMatrix::FeatureMatrix(const std::string name, const int num_rows, ConfigurationSet* config, Logger* logger)
    : num_rows_(num_rows),
      config_(config),
      feature_vector_dev_(nullptr),
      logger_(logger) {
  name_ = name;
  // Allocate memory for feature vector
  feature_vector_.assign(num_rows_ * config_->num_features, 0);
  gradient_vector_.assign(num_rows_ * config_->num_features, 0);
  feature_half_vector_.assign(num_rows_ * config_->num_features, 0);
  gradient_half_vector_.assign(num_rows_ * config_->num_features, 0);
  lock_vector_.assign(num_rows_, false);
  // Generate filename
  time_t timer = time(nullptr);
  tm local_time = *localtime(&timer);
  char time_string[1024] = {0};
  snprintf(time_string,
           sizeof time_string,
           "%02d-%02d-%02d-%02d-%02d",
           local_time.tm_mon,
           local_time.tm_mday,
           local_time.tm_hour,
           local_time.tm_min,
           local_time.tm_sec);
  filename_ = name_ + "-" + time_string + ".txt";
}

FeatureMatrix::~FeatureMatrix() {
}

int FeatureMatrix::Initialize() {
  std::default_random_engine rng;
  std::uniform_real_distribution<float> dist(0, 0.1);
  for (int i = 0; i < feature_vector_.size(); ++i) {
    VariedPrecisionFloat v(dist(rng));
    if (config_->precision == ConfigurationSet::kPrecisionHalf) {
      feature_half_vector_[i] = v.f16;
    } else if (config_->precision == ConfigurationSet::kPrecisionSingle) {
      feature_vector_[i] = v.f32;
    } else {
      assert(false);
    }
  }
  return 0;
}

int FeatureMatrix::AllocateGpuMemory() {
  hipError_t e = hipSuccess;
  if (config_->precision == ConfigurationSet::kPrecisionHalf) {
    e = hipMalloc(&feature_half_vector_dev_, feature_half_vector_.size() * sizeof feature_half_vector_[0]);
    logger_->CheckHipError(e);
    if (config_->gd_mode == ConfigurationSet::kGdModeAdapSgd) {
      e = hipMalloc(&gradient_half_vector_dev_, gradient_half_vector_.size() * sizeof gradient_half_vector_[0]);
      logger_->CheckHipError(e);
    }
  } else if (config_->precision == ConfigurationSet::kPrecisionSingle) {
    e = hipMalloc(&feature_vector_dev_, feature_vector_.size() * sizeof feature_vector_[0]);
    logger_->CheckHipError(e);
    if (config_->gd_mode == ConfigurationSet::kGdModeAdapSgd) {
      e = hipMalloc(&gradient_vector_dev_, gradient_vector_.size() * sizeof gradient_vector_[0]);
      logger_->CheckHipError(e);
    }
  } else {
    assert(false);
  }
  if (config_->lock) {
    e = hipMalloc(&lock_vector_dev_, lock_vector_.size() * sizeof lock_vector_[0]);
    logger_->CheckHipError(e);
  }
  return 0;
}

int FeatureMatrix::CopyToGpu() {
  hipError_t e = hipSuccess;
  if (config_->precision == ConfigurationSet::kPrecisionHalf) {
    e = hipMemcpy(feature_half_vector_dev_,
                  feature_half_vector_.data(),
                  feature_half_vector_.size() * sizeof feature_half_vector_[0],
                  hipMemcpyHostToDevice);
    logger_->CheckHipError(e);
    if (config_->gd_mode == ConfigurationSet::kGdModeAdapSgd) {
      e = hipMemcpy(gradient_half_vector_dev_,
                    gradient_half_vector_.data(),
                    gradient_half_vector_.size() * sizeof gradient_half_vector_[0],
                    hipMemcpyHostToDevice);
      logger_->CheckHipError(e);
    }
  } else if (config_->precision == ConfigurationSet::kPrecisionSingle) {
    e = hipMemcpy(feature_vector_dev_,
                  feature_vector_.data(),
                  feature_vector_.size() * sizeof feature_vector_[0],
                  hipMemcpyHostToDevice);
    logger_->CheckHipError(e);
    if (config_->gd_mode == ConfigurationSet::kGdModeAdapSgd) {
      e = hipMemcpy(gradient_vector_dev_,
                    gradient_vector_.data(),
                    gradient_vector_.size() * sizeof gradient_vector_[0],
                    hipMemcpyHostToDevice);
      logger_->CheckHipError(e);
    }
  } else {
    assert(false);
  }
  if (config_->lock) {
    e = hipMemcpy(lock_vector_dev_,
                  lock_vector_.data(),
                  lock_vector_.size() * sizeof lock_vector_[0],
                  hipMemcpyHostToDevice);
    logger_->CheckHipError(e);
  }
  return 0;
}

int FeatureMatrix::CopyToCpu() {
  hipError_t e = hipSuccess;
  if (config_->precision == ConfigurationSet::kPrecisionHalf) {
    e = hipMemcpy(feature_half_vector_.data(),
                  feature_half_vector_dev_,
                  feature_half_vector_.size() * sizeof feature_half_vector_[0],
                  hipMemcpyDeviceToHost);
    logger_->CheckHipError(e);
    if (config_->gd_mode == ConfigurationSet::kGdModeAdapSgd) {
      e = hipMemcpy(gradient_half_vector_.data(),
                    gradient_half_vector_dev_,
                    gradient_half_vector_.size() * sizeof gradient_half_vector_[0],
                    hipMemcpyDeviceToHost);
      logger_->CheckHipError(e);
    }
  } else if (config_->precision == ConfigurationSet::kPrecisionSingle) {
    e = hipMemcpy(feature_vector_.data(),
                  feature_vector_dev_,
                  feature_vector_.size() * sizeof feature_vector_[0],
                  hipMemcpyDeviceToHost);
    logger_->CheckHipError(e);
    if (config_->gd_mode == ConfigurationSet::kGdModeAdapSgd) {
      e = hipMemcpy(gradient_vector_.data(),
                    gradient_vector_dev_,
                    gradient_vector_.size() * sizeof gradient_vector_[0],
                    hipMemcpyDeviceToHost);
      logger_->CheckHipError(e);
    }
  } else {
    assert(false);
  }
  return 0;
}

int FeatureMatrix::DumpToFile() {
  logger_->Info(stderr, "Dumping features '%s'...\n", name_.c_str());
  FILE* f = fopen(filename_.c_str(), "w");
  for (int row = 0; row < num_rows_; ++row) {
    for (int feature_id = 0; feature_id < config_->num_features; ++feature_id) {
      if (config_->precision == ConfigurationSet::kPrecisionHalf) {
        fprintf(f, "%10.2g ", VariedPrecisionFloat(feature_half_vector_[row * config_->num_features + feature_id]).f32);
      } else if (config_->precision == ConfigurationSet::kPrecisionSingle) {
        fprintf(f, "%10.2g ", feature_vector_[row * config_->num_features + feature_id]);
      } else {
        assert(false);
      }
    }
    fprintf(f, "\n");
  }
  fclose(f);
  logger_->Info(stderr, "Features '%s' dumped to file '%s'\n", name_.c_str(), filename_.c_str());
  return 0;
}

