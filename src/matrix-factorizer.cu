#include "matrix-factorizer.h"

// C++ headers
#include <cassert>
#include <cinttypes>
#include <cstdlib>
#include <fstream>
#include <numeric>
#include <sstream>

// Linux headers
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>

// Project headers
#include "adap-sgd-feature-half-kernel.h"
#include "adap-sgd-feature-kernel.h"
#include "common.h"
#include "sgd-feature-half-kernel.h"
#include "sgd-feature-kernel.h"
#include "sgd-record-kernel.h"
#include "sgd-record-lock-kernel.h"
#include "sgd-row-kernel.h"

MatrixFactorizer::MatrixFactorizer()
    : train_dataset_(NULL), test_dataset_(NULL), user_features_(NULL),
      item_features_(NULL), train_rmse_(0), test_rmse_(0), gpu_(NULL) {
  logger_ = new Logger();
  config_ = new ConfigurationSet(logger_);
  srand(time(NULL));
}

MatrixFactorizer::~MatrixFactorizer() {
  delete gpu_;
  delete test_dataset_;
  delete train_dataset_;
  delete config_;
  delete logger_;
}

int MatrixFactorizer::Setup(
    std::unordered_map<std::string, std::string>& arg_map) {
  assert(arg_map.find("-t") != arg_map.end());
  train_dataset_ = new Dataset("train", arg_map["-t"], logger_);
  train_dataset_->Load();
  logger_->Debug(stderr, "Train dataset loaded\n");
  if (arg_map.find("-e") != arg_map.end()) {
    test_dataset_ = new Dataset("test", arg_map["-e"], logger_);
    test_dataset_->Load();
    logger_->Debug(stderr, "Test dataset loaded\n");
  }
  if (arg_map.find("-c") != arg_map.end()) {
    config_->Load(arg_map["-c"]);
    logger_->Debug(stderr, "%s\n", config_->ToString().c_str());
  }
  if (arg_map["-d"].compare("gpu") == 0) {
    gpu_ = new Gpu(0, logger_);
  }
  return 0;
}

int MatrixFactorizer::InitializeFeatures() {
  logger_->Info(stderr, "Initializing features...\n");
  user_features_ = new FeatureMatrix("user", train_dataset_->GetNumUsers(),
                                     config_, logger_);
  user_features_->Initialize();
  item_features_ = new FeatureMatrix("item", train_dataset_->GetNumItems(),
                                     config_, logger_);
  item_features_->Initialize();
  logger_->Info(stderr, "Features initialized\n");
  return 0;
}

int MatrixFactorizer::Preprocess() {
  logger_->Info(stderr, "Preprocessing data\n");
  train_dataset_->Shuffle();
  if (config_->decomp_mode == ConfigurationSet::kDecompModeRow) {
    train_dataset_->CollectUserInfo();
    train_dataset_->ShuffleUsers();
  }
  if (gpu_) {
    if (config_->decomp_mode == ConfigurationSet::kDecompModeRow) {
      train_dataset_->GenerateCooByUsers();
    } else {
      train_dataset_->GenerateCoo();
    }
    if (test_dataset_) {
      test_dataset_->GenerateCoo();
    }
    user_features_->Flatten();
    item_features_->Flatten();
  }
  logger_->Info(stderr, "Data preprocessed\n");
  return 0;
}

int MatrixFactorizer::AllocateGpuMemory() {
  logger_->Info(stderr, "Allocating GPU memory...\n");
  train_dataset_->AllocateCooGpu();
  if (config_->decomp_mode == ConfigurationSet::kDecompModeRow) {
    train_dataset_->AllocateUserInfoGpu();
  }
  user_features_->AllocateGpuMemory();
  item_features_->AllocateGpuMemory();
  logger_->Info(stderr, "GPU memory allocated\n");
  return 0;
}

int MatrixFactorizer::CopyToGpu() {
  logger_->Info(stderr, "Copying data to GPU...\n");
  train_dataset_->CopyCooToGpu();
  if (config_->decomp_mode == ConfigurationSet::kDecompModeRow) {
    train_dataset_->CopyUserInfoToGpu();
  }
  user_features_->CopyToGpu();
  item_features_->CopyToGpu();
  logger_->Info(stderr, "Data copied\n");
  return 0;
}

int MatrixFactorizer::TrainCpu() {
  if (config_->show_train_rmse) {
    ComputeTrainRmse();
  }
  if (config_->show_test_rmse && test_dataset_) {
    ComputeTestRmse();
  }
  for (int iter = 1; iter <= config_->max_num_iterations; ++iter) {
    logger_->Info(stderr, "Starting iteration %-4d\n", iter);
    switch (config_->gd_mode) {
    case ConfigurationSet::kGdModeMiniBatchSgd:
      if (config_->batch_size == 1) {
        RunSgdFeature();
      } else {
        RunMiniBatchSgdFeature();
      }
      break;
    case ConfigurationSet::kGdModeAdapSgd:
      RunAdapSgdFeature();
      break;
    default: assert(false);
    }
    logger_->StopTimer();
    logger_->Info(stderr, "Kernel time = %-8g\n", logger_->ReadTimer());
    if (config_->show_train_rmse) {
      ComputeTrainRmse();
    }
    if (config_->show_test_rmse && test_dataset_) {
      ComputeTestRmse();
    }
  }
  return 0;
}

int MatrixFactorizer::RunSgdFeature() {
  logger_->Info(stderr, "Running SGD...\n");
  logger_->StartTimer();
  for (int record_id = 0; record_id < train_dataset_->GetNumRecords();
       ++record_id) {
    // Compute error
    const int user_id = train_dataset_->GetUserId(record_id);
    const int item_id = train_dataset_->GetItemId(record_id);
    const auto& user_feature_vector = user_features_->
        GetFeatureVector(user_id);
    const auto& item_feature_vector = item_features_->
        GetFeatureVector(item_id);
    const float predicted_rating = std::inner_product(
        user_feature_vector.begin(), user_feature_vector.end(),
        item_feature_vector.begin(), 0.0);
    const float error = predicted_rating -
        train_dataset_->GetRating(record_id);
    // Update features
    std::vector<float> new_user_feature_vector(config_->num_features, 0);
    std::vector<float> new_item_feature_vector(config_->num_features, 0);
    for (int feature_id = 0; feature_id < config_->num_features;
         ++feature_id) {
      const float user_feature = user_feature_vector[feature_id];
      const float item_feature = item_feature_vector[feature_id];
      const float user_grad = config_->regularization_factor * user_feature +
          error * item_feature;
      const float item_grad = config_->regularization_factor * item_feature +
          error * user_feature;
      new_user_feature_vector[feature_id] = user_feature -
          config_->learning_rate * user_grad;
      new_item_feature_vector[feature_id] = item_feature -
          config_->learning_rate * item_grad;
    }
    user_features_->SetFeatureVector(user_id, new_user_feature_vector);
    item_features_->SetFeatureVector(item_id, new_item_feature_vector);
  }
  logger_->Info(stderr, "SGD done\n");
  return 0;
}

int MatrixFactorizer::RunMiniBatchSgdFeature() {
  logger_->Info(stderr, "Running mini-batch SGD...\n");
  logger_->StartTimer();
  const int num_batches = (train_dataset_->GetNumRecords() +
      config_->batch_size - 1) / config_->batch_size;
  std::vector<float> error_vector(config_->batch_size, 0);
  std::map<int, std::vector<float>> user_grad_map;
  std::map<int, std::vector<float>> item_grad_map;
  for (int batch_id = 0; batch_id < num_batches; ++batch_id) {
    // Compute gradients
    for (int record_offset = 0; record_offset < config_->batch_size;
         ++record_offset) {
      const int record_id = batch_id * config_->batch_size + record_offset;
      if (record_id >= train_dataset_->GetNumRecords()) {
        break;
      }
      const int user_id = train_dataset_->GetUserId(record_id);
      const int item_id = train_dataset_->GetItemId(record_id);
      const auto& user_feature_vector =
          user_features_->GetFeatureVector(user_id);
      const auto& item_feature_vector =
          item_features_->GetFeatureVector(item_id);
      error_vector[record_offset] = std::inner_product(
          user_feature_vector.begin(), user_feature_vector.end(),
          item_feature_vector.begin(), 0.0) -
          train_dataset_->GetRating(record_id);
      if (user_grad_map.find(user_id) == user_grad_map.end()) {
        user_grad_map[user_id].assign(config_->num_features, 0);
      }
      if (item_grad_map.find(item_id) == item_grad_map.end()) {
        item_grad_map[item_id].assign(config_->num_features, 0);
      }
      for (int feature_id = 0; feature_id < config_->num_features;
           ++feature_id) {
        user_grad_map[user_id][feature_id] += error_vector[record_offset] *
            item_feature_vector[feature_id];
        item_grad_map[item_id][feature_id] += error_vector[record_offset] *
            user_feature_vector[feature_id];
      }
    }
    // Update features
    for (const auto& id_grad : user_grad_map) {
      const int user_id = id_grad.first;
      auto user_feature_vector = user_features_->GetFeatureVector(user_id);
      for (int feature_id = 0; feature_id < config_->num_features;
           ++feature_id) {
        float user_feature = user_feature_vector[feature_id];
        user_feature -= config_->learning_rate * (id_grad.second[feature_id] +
            config_->regularization_factor * user_feature);
        user_feature_vector[feature_id] = user_feature;
      }
      user_features_->SetFeatureVector(user_id, user_feature_vector);
    }
    for (const auto& id_grad : item_grad_map) {
      const int item_id = id_grad.first;
      auto item_feature_vector = item_features_->GetFeatureVector(item_id);
      for (int feature_id = 0; feature_id < config_->num_features;
           ++feature_id) {
        float item_feature = item_feature_vector[feature_id];
        item_feature -= config_->learning_rate * (id_grad.second[feature_id] +
            config_->regularization_factor * item_feature);
        item_feature_vector[feature_id] = item_feature;
      }
      item_features_->SetFeatureVector(item_id, item_feature_vector);
    }
    // Clear containers
    for (auto& id_grad : user_grad_map) {
      id_grad.second.clear();
    }
    for (auto& id_grad : item_grad_map) {
      id_grad.second.clear();
    }
    user_grad_map.clear();
    item_grad_map.clear();
  }
  logger_->Info(stderr, "Mini-batch SGD done\n");
  return 0;
}

int MatrixFactorizer::RunAdapSgdFeature() {
  logger_->Info(stderr, "Running adaptive SGD...\n");
  logger_->StartTimer();
  for (int record_id = 0; record_id < train_dataset_->GetNumRecords();
       ++record_id) {
    // Compute error
    const int user_id = train_dataset_->GetUserId(record_id);
    const int item_id = train_dataset_->GetItemId(record_id);
    const auto& user_feature_vector = user_features_->
        GetFeatureVector(user_id);
    const auto& item_feature_vector = item_features_->
        GetFeatureVector(item_id);
    const float predicted_rating = std::inner_product(
        user_feature_vector.begin(), user_feature_vector.end(),
        item_feature_vector.begin(), 0.0);
    const float error = predicted_rating -
        train_dataset_->GetRating(record_id);
    // Update features
    const auto& user_grad_vector = user_features_->GetGradientVector(user_id);
    std::vector<float> new_user_feature_vector(config_->num_features, 0);
    std::vector<float> new_user_grad_vector(config_->num_features, 0);
    const auto& item_grad_vector = item_features_->GetGradientVector(item_id);
    std::vector<float> new_item_feature_vector(config_->num_features, 0);
    std::vector<float> new_item_grad_vector(config_->num_features, 0);
    for (int feature_id = 0; feature_id < config_->num_features;
         ++feature_id) {
      const float user_feature = user_feature_vector[feature_id];
      const float item_feature = item_feature_vector[feature_id];
      const float user_grad = config_->regularization_factor * user_feature +
          error * item_feature;
      const float item_grad = config_->regularization_factor * item_feature +
          error * user_feature;
      const float new_user_grad = user_grad_vector[feature_id] +
          user_grad * user_grad;
      const float new_item_grad = item_grad_vector[feature_id] +
          item_grad * item_grad;
      const float new_user_learning_rate = config_->learning_rate /
          sqrt(new_user_grad);
      const float new_item_learning_rate = config_->learning_rate /
          sqrt(new_item_grad);
      new_user_feature_vector[feature_id] = user_feature -
          new_user_learning_rate * user_grad;
      new_item_feature_vector[feature_id] = item_feature -
          new_item_learning_rate * item_grad;
      new_user_grad_vector[feature_id] = new_user_grad;
      new_item_grad_vector[feature_id] = new_item_grad;
    }
    user_features_->SetFeatureVector(user_id, new_user_feature_vector);
    item_features_->SetFeatureVector(item_id, new_item_feature_vector);
    user_features_->SetGradientVector(user_id, new_user_grad_vector);
    item_features_->SetGradientVector(item_id, new_item_grad_vector);
  }
  logger_->Info(stderr, "Adaptive SGD done\n");
  return 0;
}

int MatrixFactorizer::TrainGpu() {
  if (config_->show_train_rmse) {
    ComputeTrainRmse();
  }
  if (config_->show_test_rmse && test_dataset_) {
    ComputeTestRmse();
  }
  for (int iter = 1; iter <= config_->max_num_iterations; ++iter) {
    logger_->Info(stderr, "Starting iteration %-4d\n", iter);
    switch (config_->gd_mode) {
    case ConfigurationSet::kGdModeMiniBatchSgd:
      if (config_->batch_size == 1) {
        if (config_->decomp_mode == ConfigurationSet::kDecompModeRecord) {
          if (config_->lock) {
            LaunchSgdRecordLockKernel();
          } else {
            LaunchSgdRecordKernel();
          }
        } else if (config_->decomp_mode == ConfigurationSet::kDecompModeRow) {
          LaunchSgdRowKernel();
        } else if (config_->decomp_mode ==
            ConfigurationSet::kDecompModeFeature) {
          if (config_->precision == ConfigurationSet::kPrecisionHalf) {
            LaunchSgdFeatureHalfKernel();
          } else if (config_->precision ==
                         ConfigurationSet::kPrecisionSingle) {
            LaunchSgdFeatureKernel();
          } else if (config_->precision ==
                         ConfigurationSet::kPrecisionDouble) {
            assert(false);
          }
        }
      } else {
        assert(false);
      }
      break;
    case ConfigurationSet::kGdModeAdapSgd:
      if (config_->precision == ConfigurationSet::kPrecisionHalf) {
        LaunchAdapSgdFeatureHalfKernel();
      } else if (config_->precision == ConfigurationSet::kPrecisionSingle) {
        LaunchAdapSgdFeatureKernel();
      } else if (config_->precision == ConfigurationSet::kPrecisionDouble) {
        assert(false);
      }
      break;
    default: assert(false);
    }
    cudaError_t e = cudaSuccess;
    e = cudaStreamSynchronize(gpu_->GetShaderStream());
    logger_->CheckCudaError(e);
    logger_->StopTimer();
    logger_->Info(stderr, "Kernel time = %-8g\n", logger_->ReadTimer());
    if (config_->show_train_rmse ||
        (config_->show_test_rmse && test_dataset_)) {
      user_features_->CopyToCpuFlatten();
      item_features_->CopyToCpuFlatten();
    }
    if (config_->show_train_rmse) {
      ComputeTrainRmse();
    }
    if (config_->show_test_rmse && test_dataset_) {
      ComputeTestRmse();
    }
  }
  return 0;
}

int MatrixFactorizer::LaunchSgdRecordLockKernel() {
  cudaError_t e = cudaSuccess;
  e = cudaStreamSynchronize(gpu_->GetH2dStream());
  logger_->CheckCudaError(e);
  const int num_records = train_dataset_->GetNumRecords();
  dim3 block_size(128, 1, 1);
  dim3 grid_size((num_records + block_size.x - 1) / block_size.x, 1, 1);
  logger_->Info(stderr, "Launching kernels...\n");
  logger_->StartTimer();
  SgdRecordLock<<<grid_size, block_size, 0, gpu_->GetShaderStream()>>>(
      train_dataset_->GetUserIdDev(), train_dataset_->GetItemIdDev(),
      train_dataset_->GetRatingDev(), num_records, config_->num_features,
      config_->learning_rate, config_->regularization_factor,
      user_features_->GetLockDev(), item_features_->GetLockDev(),
      user_features_->GetFeatureDev(), item_features_->GetFeatureDev());
  logger_->Info(stderr, "Kernels launched\n");
  return 0;
}

int MatrixFactorizer::LaunchSgdRecordKernel() {
  cudaError_t e = cudaSuccess;
  e = cudaStreamSynchronize(gpu_->GetH2dStream());
  logger_->CheckCudaError(e);
  const int num_records = train_dataset_->GetNumRecords();
  dim3 block_size(128, 1, 1);
  dim3 grid_size((num_records + block_size.x - 1) / block_size.x, 1, 1);
  logger_->Info(stderr, "Launching kernels...\n");
  logger_->StartTimer();
  SgdRecord<<<grid_size, block_size, 0, gpu_->GetShaderStream()>>>(
      train_dataset_->GetUserIdDev(), train_dataset_->GetItemIdDev(),
      train_dataset_->GetRatingDev(), num_records, config_->num_features,
      config_->learning_rate, config_->regularization_factor,
      user_features_->GetFeatureDev(), item_features_->GetFeatureDev());
  logger_->Info(stderr, "Kernels launched\n");
  return 0;
}

int MatrixFactorizer::LaunchSgdRowKernel() {
  cudaError_t e = cudaSuccess;
  e = cudaStreamSynchronize(gpu_->GetH2dStream());
  logger_->CheckCudaError(e);
  const int num_users = train_dataset_->GetNumUsers();
  dim3 block_size(128, 1, 1);
  dim3 grid_size((num_users + block_size.x - 1) / block_size.x, 1, 1);
  logger_->Info(stderr, "Launching kernels...\n");
  logger_->StartTimer();
  SgdRow<<<grid_size, block_size, 0, gpu_->GetShaderStream()>>>(
      train_dataset_->GetUserIdDev(), train_dataset_->GetItemIdDev(),
      train_dataset_->GetRatingDev(), num_users,
      train_dataset_->GetUserRecordBaseDev(),
      train_dataset_->GetUserNumRecordsDev(), config_->num_features,
      config_->learning_rate, config_->regularization_factor,
      user_features_->GetFeatureDev(), item_features_->GetFeatureDev());
  logger_->Info(stderr, "Kernels launched\n");
  return 0;
}

int MatrixFactorizer::LaunchSgdFeatureHalfKernel() {
  cudaError_t e = cudaSuccess;
  e = cudaStreamSynchronize(gpu_->GetH2dStream());
  logger_->CheckCudaError(e);
  dim3 grid_size(train_dataset_->GetNumRecords(), 1, 1);
  dim3 block_size(config_->num_features, 1, 1);
  logger_->Info(stderr, "Launching kernels...\n");
  logger_->StartTimer();
  SgdFeatureHalf<<<grid_size, block_size,
      config_->num_features * sizeof(float), gpu_->GetShaderStream()>>>(
      train_dataset_->GetUserIdDev(), train_dataset_->GetItemIdDev(),
      train_dataset_->GetRatingDev(), config_->num_features,
      config_->learning_rate, config_->regularization_factor,
      user_features_->GetFeatureHalfDev(), item_features_->GetFeatureHalfDev());
  logger_->Info(stderr, "Kernels launched\n");
  return 0;
}

int MatrixFactorizer::LaunchSgdFeatureKernel() {
  cudaError_t e = cudaSuccess;
  e = cudaStreamSynchronize(gpu_->GetH2dStream());
  logger_->CheckCudaError(e);
  dim3 grid_size(train_dataset_->GetNumRecords(), 1, 1);
  dim3 block_size(config_->num_features, 1, 1);
  logger_->Info(stderr, "Launching kernels...\n");
  logger_->StartTimer();
  SgdFeature<<<grid_size, block_size,
      config_->num_features * sizeof(float), gpu_->GetShaderStream()>>>(
      train_dataset_->GetUserIdDev(), train_dataset_->GetItemIdDev(),
      train_dataset_->GetRatingDev(), config_->num_features,
      config_->learning_rate, config_->regularization_factor,
      user_features_->GetFeatureDev(), item_features_->GetFeatureDev());
  logger_->Info(stderr, "Kernels launched\n");
  return 0;
}

int MatrixFactorizer::LaunchAdapSgdFeatureHalfKernel() {
  cudaError_t e = cudaSuccess;
  e = cudaStreamSynchronize(gpu_->GetH2dStream());
  logger_->CheckCudaError(e);
  dim3 grid_size(train_dataset_->GetNumRecords(), 1, 1);
  dim3 block_size(config_->num_features, 1, 1);
  logger_->Info(stderr, "Launching kernels...\n");
  logger_->StartTimer();
  AdapSgdFeatureHalf<<<grid_size, block_size,
      config_->num_features * sizeof(float), gpu_->GetShaderStream()>>>(
      train_dataset_->GetUserIdDev(), train_dataset_->GetItemIdDev(),
      train_dataset_->GetRatingDev(), config_->num_features,
      config_->learning_rate, config_->regularization_factor,
      user_features_->GetParameterDev(), item_features_->GetParameterDev());
  logger_->Info(stderr, "Kernels launched\n");
  return 0;
}

int MatrixFactorizer::LaunchAdapSgdFeatureKernel() {
  cudaError_t e = cudaSuccess;
  e = cudaStreamSynchronize(gpu_->GetH2dStream());
  logger_->CheckCudaError(e);
  dim3 grid_size(train_dataset_->GetNumRecords(), 1, 1);
  dim3 block_size(config_->num_features, 1, 1);
  logger_->Info(stderr, "Launching kernels...\n");
  logger_->StartTimer();
  AdapSgdFeature<<<grid_size, block_size,
      config_->num_features * sizeof(float), gpu_->GetShaderStream()>>>(
      train_dataset_->GetUserIdDev(), train_dataset_->GetItemIdDev(),
      train_dataset_->GetRatingDev(), config_->num_features,
      config_->learning_rate, config_->regularization_factor,
      user_features_->GetFeatureDev(), item_features_->GetFeatureDev(),
      user_features_->GetGradientDev(), item_features_->GetGradientDev());
  logger_->Info(stderr, "Kernels launched\n");
  return 0;
}

int MatrixFactorizer::ComputeTrainRmse() {
  logger_->Info(stderr, "Computing training rmse...\n");
  float error = 0;
  error = ComputeSquareErrorSum(train_dataset_);
  train_rmse_ = sqrt(error / train_dataset_->GetNumRecords());
  logger_->Info(stderr, "Training rmse = %f\n", train_rmse_);
  return 0;
}

int MatrixFactorizer::ComputeTestRmse() {
  logger_->Info(stderr, "Computing testing rmse...\n");
  float error = 0;
  error = ComputeSquareErrorSum(test_dataset_);
  test_rmse_ = sqrt(error / test_dataset_->GetNumRecords());
  logger_->Info(stderr, "Testing rmse = %f\n", test_rmse_);
  return 0;
}

int MatrixFactorizer::DumpFeatures() {
  if (!gpu_) {
    user_features_->DumpToFile();
    item_features_->DumpToFile();
    return 0;
  }
  user_features_->CopyToCpuFlatten();
  item_features_->CopyToCpuFlatten();
  user_features_->Dump1dToFile();
  item_features_->Dump1dToFile();
  return 0;
}

float MatrixFactorizer::ComputeSquareErrorSum(Dataset* dataset) {
  float error = 0;
  float c = 0;
  for (int record_id = 0; record_id < dataset->GetNumRecords();
       ++record_id) {
    const int user_id = dataset->GetUserId(record_id);
    const int item_id = dataset->GetItemId(record_id);
    float e = 0;
    e = PredictRating(user_id, item_id) - dataset->GetRating(record_id);
    e *= e;
    // Kahan sum algorithm
    float y = e - c;
    float t = error + y;
    c = t - error - y;
    error = t;
  }
  return error;
}

float MatrixFactorizer::PredictRating(const int64_t user_id,
                                      const int64_t item_id) {
  if (!gpu_) {
    const auto& user_feature_vector =
        user_features_->GetFeatureVector(user_id);
    const auto& item_feature_vector =
        item_features_->GetFeatureVector(item_id);
    return std::inner_product(user_feature_vector.begin(),
                              user_feature_vector.end(),
                              item_feature_vector.begin(), 0.0);
  }
  if (config_->precision == ConfigurationSet::kPrecisionHalf) {
    if (config_->gd_mode == ConfigurationSet::kGdModeMiniBatchSgd) {
      const auto& user_feature =
          user_features_->GetFeatureHalf1dVector().data();
      const auto& item_feature =
          item_features_->GetFeatureHalf1dVector().data();
      return std::inner_product(
          user_feature + user_id * config_->num_features,
          user_feature + (user_id + 1) * config_->num_features,
          item_feature + item_id * config_->num_features, 0.0,
          std::plus<double>(),
          [](unsigned short a, unsigned short b) {
            return Half2Float(a) * Half2Float(b);
          });
    } else if (config_->gd_mode == ConfigurationSet::kGdModeAdapSgd) {
      const auto& user_parameters =
          user_features_->GetParameter1dVector().data();
      const auto& item_parameters =
          item_features_->GetParameter1dVector().data();
      return std::inner_product(
          user_parameters + user_id * config_->num_features,
          user_parameters + (user_id + 1) * config_->num_features,
          item_parameters + item_id * config_->num_features, 0.0,
          std::plus<double>(),
          [](unsigned int a, unsigned int b) {
            return Half2Float(a & 0xffff) * Half2Float(b & 0xffff);
          });
    }
  } else if (config_->precision == ConfigurationSet::kPrecisionSingle) {
    const auto& user_features = user_features_->GetFeature1dVector().data();
    const auto& item_features = item_features_->GetFeature1dVector().data();
    return std::inner_product(
        user_features + user_id * config_->num_features,
        user_features + (user_id + 1) * config_->num_features,
        item_features + item_id * config_->num_features, 0.0);
  }
  return 0;
}

