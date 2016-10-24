#ifndef MATRIX_FACTORIZER_H_
#define MATRIX_FACTORIZER_H_

// C++ headers
#include <unordered_map>

// Project headers
#include "configuration-set.h"
#include "dataset.h"
#include "feature-matrix.h"
#include "gpu.h"
#include "logger.h"
#include "record.h"
#include "user.h"

class MatrixFactorizer {
 public:
  MatrixFactorizer();
  ~MatrixFactorizer();
  // Setup datasets, configurations, etc.
  int Setup(std::unordered_map<std::string, std::string>& arg_map);
  // Initialization
  int InitializeFeatures();
  // Preprocessing
  int Preprocess();
  // GPU memory
  int AllocateGpuMemory();
  int CopyToGpu();
  // CPU training
  int TrainCpu();
  int RunSgdFeature();
  int RunMiniBatchSgdFeature();
  int RunAdapSgdFeature();
  // GPU training
  int TrainGpu();
  int LaunchSgdRecordLockKernel();
  int LaunchSgdRecordKernel();
  int LaunchSgdRowKernel();
  int LaunchSgdFeatureHalfKernel();
  int LaunchSgdFeatureKernel();
  int LaunchAdapSgdFeatureHalfKernel();
  int LaunchAdapSgdFeatureKernel();
  // Statistics
  int ComputeTrainRmse();
  int ComputeTestRmse();
  // Output
  int DumpFeatures();
  // Helpers
  float ComputeSquareErrorSum(Dataset* dataset);
  float PredictRating(const int64_t user_id, const int64_t item_id);
 private:
  // Datasets
  Dataset* train_dataset_;
  Dataset* test_dataset_;
  // Features
  FeatureMatrix* user_features_;
  FeatureMatrix* item_features_;
  // Configurations
  ConfigurationSet* config_;
  // Statistics
  float train_rmse_;
  float test_rmse_;
  // GPU
  Gpu* gpu_;
  // Logger
  Logger* logger_;
};

#endif

