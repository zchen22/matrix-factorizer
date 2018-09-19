#include "matrix-factorizer.h"

// C++ headers
#include <cassert>
#include <cinttypes>
#include <cstdlib>
#include <fstream>
#include <numeric>
#include <sstream>
#include <thread>

// Linux headers
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>

// HIP headers
#include "hip/hip_runtime.h"

// Project headers
#include "kernels/adap-sgd-feature-half-kernel.h"
#include "kernels/adap-sgd-feature-kernel.h"
#include "kernels/sgd-feature-half-kernel.h"
#include "kernels/sgd-feature-kernel.h"
#include "kernels/sgd-record-kernel.h"
#include "kernels/sgd-record-lock-kernel.h"
#include "kernels/sgd-row-kernel.h"
#include "varied-precision-float.h"

MatrixFactorizer::MatrixFactorizer()
    : train_dataset_(nullptr),
      test_dataset_(nullptr),
      user_features_(nullptr),
      item_features_(nullptr),
      train_rmse_(0),
      test_rmse_(0),
      gpu_(nullptr) {
  logger_ = new Logger();
  config_ = new ConfigurationSet(logger_);
  const uint32_t seed = time(nullptr);
  logger_->Debug(stderr, "Seed = %u\n", seed);
  srand(seed);
}

MatrixFactorizer::~MatrixFactorizer() {
  delete config_;
  delete logger_;
  delete gpu_;
  delete test_dataset_;
  delete train_dataset_;
}

int MatrixFactorizer::Setup(std::unordered_map<std::string, std::string>& arg_map) {
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
  if (arg_map.find("-d") != arg_map.end()) {
    gpu_ = new Gpu(std::stoi(arg_map["-d"]), logger_);
  }
  return 0;
}

int MatrixFactorizer::InitializeFeatures() {
  logger_->Info(stderr, "Initializing features...\n");
  user_features_ = new FeatureMatrix("user", train_dataset_->GetNumUsers(), config_, logger_);
  user_features_->Initialize();
  item_features_ = new FeatureMatrix("item", train_dataset_->GetNumItems(), config_, logger_);
  item_features_->Initialize();
  logger_->Info(stderr, "Features initialized\n");
  return 0;
}

int MatrixFactorizer::Preprocess() {
  logger_->Info(stderr, "Preprocessing data\n");
  train_dataset_->Shuffle();
  if (config_->decomp_mode == ConfigurationSet::kDecompModeRow) {
    train_dataset_->CollectUserInfo();
    train_dataset_->ShuffleByUser();
  }
  if (gpu_) {
    train_dataset_->GenerateCoo();
    if (config_->decomp_mode == ConfigurationSet::kDecompModeRow) {
      train_dataset_->GenerateCooByUser();
    }
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
    logger_->StartTimer();
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
  for (int record_id = 0; record_id < train_dataset_->GetNumRecords(); ++record_id) {
    // Compute error
    const int user_id = train_dataset_->GetUserId(record_id);
    const int item_id = train_dataset_->GetItemId(record_id);
    const float error = PredictRating(user_id, item_id) - train_dataset_->GetRating(record_id);
    // Update features
    for (int feature_id = 0; feature_id < config_->num_features; ++feature_id) {
      const float user_feature = user_features_->GetFeatureVector()[user_id * config_->num_features + feature_id];
      const float item_feature = item_features_->GetFeatureVector()[item_id * config_->num_features + feature_id];
      const float user_grad = config_->regularization_factor.f32 * user_feature + error * item_feature;
      const float item_grad = config_->regularization_factor.f32 * item_feature + error * user_feature;
      user_features_->SetFeature(user_id, feature_id, user_feature - config_->learning_rate.f32 * user_grad);
      item_features_->SetFeature(item_id, feature_id, item_feature - config_->learning_rate.f32 * item_grad);
    }
  }
  logger_->Info(stderr, "SGD done\n");
  return 0;
}

int MatrixFactorizer::RunMiniBatchSgdFeature() {
  logger_->Info(stderr, "Running mini-batch SGD...\n");
  const int num_batches = (train_dataset_->GetNumRecords() + config_->batch_size - 1) / config_->batch_size;
  std::vector<float> error_vector(config_->batch_size, 0);
  for (int batch_id = 0; batch_id < num_batches; ++batch_id) {
    std::unordered_map<int, std::vector<float>> user_grad_map;
    std::unordered_map<int, std::vector<float>> item_grad_map;
    // Compute gradients
    for (int record_offset = 0; record_offset < config_->batch_size; ++record_offset) {
      const int record_id = batch_id * config_->batch_size + record_offset;
      if (record_id >= train_dataset_->GetNumRecords()) {
        break;
      }
      const int user_id = train_dataset_->GetUserId(record_id);
      const int item_id = train_dataset_->GetItemId(record_id);
      error_vector[record_offset] = PredictRating(user_id, item_id) - train_dataset_->GetRating(record_id);
      if (user_grad_map.find(user_id) == user_grad_map.end()) {
        user_grad_map[user_id].assign(config_->num_features, 0);
      }
      if (item_grad_map.find(item_id) == item_grad_map.end()) {
        item_grad_map[item_id].assign(config_->num_features, 0);
      }
      for (int feature_id = 0; feature_id < config_->num_features; ++feature_id) {
        user_grad_map[user_id][feature_id] += error_vector[record_offset] * item_features_->GetFeatureVector()[item_id * config_->num_features + feature_id];
        item_grad_map[item_id][feature_id] += error_vector[record_offset] * user_features_->GetFeatureVector()[user_id * config_->num_features + feature_id];
      }
    }
    // Update features
    for (const auto& id_grad : user_grad_map) {
      const int user_id = id_grad.first;
      for (int feature_id = 0; feature_id < config_->num_features; ++feature_id) {
        float user_feature = user_features_->GetFeatureVector()[user_id * config_->num_features + feature_id];
        user_feature -= config_->learning_rate.f32 * (id_grad.second[feature_id] + config_->regularization_factor.f32 * user_feature);
        user_features_->SetFeature(user_id, feature_id, user_feature);
      }
    }
    for (const auto& id_grad : item_grad_map) {
      const int item_id = id_grad.first;
      for (int feature_id = 0; feature_id < config_->num_features; ++feature_id) {
        float item_feature = item_features_->GetFeatureVector()[item_id * config_->num_features + feature_id];
        item_feature -= config_->learning_rate.f32 * (id_grad.second[feature_id] + config_->regularization_factor.f32 * item_feature);
        item_features_->SetFeature(item_id, feature_id, item_feature);
      }
    }
  }
  logger_->Info(stderr, "Mini-batch SGD done\n");
  return 0;
}

int MatrixFactorizer::RunAdapSgdFeature() {
  logger_->Info(stderr, "Running adaptive SGD...\n");
  for (int record_id = 0; record_id < train_dataset_->GetNumRecords(); ++record_id) {
    // Compute error
    const int user_id = train_dataset_->GetUserId(record_id);
    const int item_id = train_dataset_->GetItemId(record_id);
    const float error = PredictRating(user_id, item_id) - train_dataset_->GetRating(record_id);
    // Update features
    for (int feature_id = 0; feature_id < config_->num_features; ++feature_id) {
      const float user_feature = user_features_->GetFeatureVector()[user_id * config_->num_features + feature_id];
      const float item_feature = item_features_->GetFeatureVector()[item_id * config_->num_features + feature_id];
      const float user_grad = config_->regularization_factor.f32 * user_feature + error * item_feature;
      const float item_grad = config_->regularization_factor.f32 * item_feature + error * user_feature;
      const float new_user_grad = user_features_->GetGradientVector()[user_id * config_->num_features + feature_id] + user_grad * user_grad;
      const float new_item_grad = item_features_->GetGradientVector()[item_id * config_->num_features + feature_id] + item_grad * item_grad;
      const float new_user_learning_rate = config_->learning_rate.f32 / sqrt(new_user_grad);
      const float new_item_learning_rate = config_->learning_rate.f32 / sqrt(new_item_grad);
      user_features_->SetFeature(user_id, feature_id, user_feature - new_user_learning_rate * user_grad);
      item_features_->SetFeature(item_id, feature_id, item_feature - new_item_learning_rate * item_grad);
      user_features_->SetGradient(user_id, feature_id, new_user_grad);
      item_features_->SetGradient(item_id, feature_id, new_item_grad);
    }
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
        } else if (config_->decomp_mode == ConfigurationSet::kDecompModeFeature) {
          if (config_->precision == ConfigurationSet::kPrecisionHalf) {
            LaunchSgdFeatureHalfKernel();
          } else if (config_->precision == ConfigurationSet::kPrecisionSingle) {
            LaunchSgdFeatureKernel();
          } else {
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
      } else {
        assert(false);
      }
      break;
    default: assert(false);
    }
    hipError_t e = hipStreamSynchronize(gpu_->shader_stream);
    logger_->CheckHipError(e);
    logger_->StopTimer();
    logger_->Info(stderr, "Kernel time = %-8g\n", logger_->ReadTimer());
    if (config_->show_train_rmse || (config_->show_test_rmse && test_dataset_)) {
      user_features_->CopyToCpu();
      item_features_->CopyToCpu();
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
  hipError_t e = hipSuccess;
  e = hipStreamSynchronize(gpu_->h2d_stream);
  logger_->CheckHipError(e);
  const int num_records = train_dataset_->GetNumRecords();
  dim3 block_size(128, 1, 1);
  dim3 grid_size((num_records + block_size.x - 1) / block_size.x, 1, 1);
  logger_->Info(stderr, "Launching kernels...\n");
  logger_->StartTimer();
  hipLaunchKernelGGL((SgdRecordLock),
                     dim3(grid_size),
                     dim3(block_size),
                     0,
                     gpu_->shader_stream,
                     train_dataset_->GetUserIdDev(),
                     train_dataset_->GetItemIdDev(),
                     train_dataset_->GetRatingDev(),
                     num_records,
                     config_->num_features,
                     config_->learning_rate.f32,
                     config_->regularization_factor.f32,
                     user_features_->GetLockDev(),
                     item_features_->GetLockDev(),
                     user_features_->GetFeatureDev(),
                     item_features_->GetFeatureDev());
  logger_->Info(stderr, "Kernels launched\n");
  return 0;
}

int MatrixFactorizer::LaunchSgdRecordKernel() {
  hipError_t e = hipSuccess;
  e = hipStreamSynchronize(gpu_->h2d_stream);
  logger_->CheckHipError(e);
  const int num_records = train_dataset_->GetNumRecords();
  dim3 block_size(128, 1, 1);
  dim3 grid_size((num_records + block_size.x - 1) / block_size.x, 1, 1);
  logger_->Info(stderr, "Launching kernels...\n");
  logger_->StartTimer();
  hipLaunchKernelGGL((SgdRecord),
                     dim3(grid_size),
                     dim3(block_size),
                     0,
                     gpu_->shader_stream,
                     train_dataset_->GetUserIdDev(),
                     train_dataset_->GetItemIdDev(),
                     train_dataset_->GetRatingDev(),
                     num_records,
                     config_->num_features,
                     config_->learning_rate.f32,
                     config_->regularization_factor.f32,
                     user_features_->GetFeatureDev(),
                     item_features_->GetFeatureDev());
  logger_->Info(stderr, "Kernels launched\n");
  return 0;
}

int MatrixFactorizer::LaunchSgdRowKernel() {
  hipError_t e = hipSuccess;
  e = hipStreamSynchronize(gpu_->h2d_stream);
  logger_->CheckHipError(e);
  const int num_users = train_dataset_->GetNumUsers();
  dim3 block_size(128, 1, 1);
  dim3 grid_size((num_users + block_size.x - 1) / block_size.x, 1, 1);
  logger_->Info(stderr, "Launching kernels...\n");
  logger_->StartTimer();
  hipLaunchKernelGGL((SgdRow),
                     dim3(grid_size),
                     dim3(block_size),
                     0,
                     gpu_->shader_stream,
                     train_dataset_->GetUserIdDev(),
                     train_dataset_->GetItemIdDev(),
                     train_dataset_->GetRatingDev(),
                     num_users,
                     train_dataset_->GetUserRecordBaseDev(),
                     train_dataset_->GetUserNumRecordsDev(),
                     config_->num_features,
                     config_->learning_rate.f32,
                     config_->regularization_factor.f32,
                     user_features_->GetFeatureDev(),
                     item_features_->GetFeatureDev());
  logger_->Info(stderr, "Kernels launched\n");
  return 0;
}

int MatrixFactorizer::LaunchSgdFeatureHalfKernel() {
  hipError_t e = hipSuccess;
  e = hipStreamSynchronize(gpu_->h2d_stream);
  logger_->CheckHipError(e);
  dim3 grid_size(train_dataset_->GetNumRecords(), 1, 1);
  dim3 block_size(config_->num_features, 1, 1);
  logger_->Info(stderr, "Launching kernels...\n");
  logger_->StartTimer();
  hipLaunchKernelGGL((SgdFeatureHalf),
                     dim3(grid_size),
                     dim3(block_size),
                     config_->num_features * sizeof(int16_t),
                     gpu_->shader_stream,
                     train_dataset_->GetUserIdDev(),
                     train_dataset_->GetItemIdDev(),
                     train_dataset_->GetRatingHalfDev(),
                     config_->num_features,
                     config_->learning_rate.f16,
                     config_->regularization_factor.f16,
                     user_features_->GetFeatureHalfDev(),
                     item_features_->GetFeatureHalfDev());
  logger_->Info(stderr, "Kernels launched\n");
  return 0;
}

int MatrixFactorizer::LaunchSgdFeatureKernel() {
  hipError_t e = hipSuccess;
  e = hipStreamSynchronize(gpu_->h2d_stream);
  logger_->CheckHipError(e);
  dim3 grid_size(train_dataset_->GetNumRecords(), 1, 1);
  dim3 block_size(config_->num_features, 1, 1);
  logger_->Info(stderr, "Launching kernels...\n");
  logger_->StartTimer();
  hipLaunchKernelGGL((SgdFeature),
                     dim3(grid_size),
                     dim3(block_size),
                     config_->num_features * sizeof(float),
                     gpu_->shader_stream,
                     train_dataset_->GetUserIdDev(),
                     train_dataset_->GetItemIdDev(),
                     train_dataset_->GetRatingDev(),
                     config_->num_features,
                     config_->learning_rate.f32,
                     config_->regularization_factor.f32,
                     user_features_->GetFeatureDev(),
                     item_features_->GetFeatureDev());
  logger_->Info(stderr, "Kernels launched\n");
  return 0;
}

int MatrixFactorizer::LaunchAdapSgdFeatureHalfKernel() {
  hipError_t e = hipSuccess;
  e = hipStreamSynchronize(gpu_->h2d_stream);
  logger_->CheckHipError(e);
  dim3 grid_size(train_dataset_->GetNumRecords(), 1, 1);
  dim3 block_size(config_->num_features, 1, 1);
  logger_->Info(stderr, "Launching kernels...\n");
  logger_->StartTimer();
  hipLaunchKernelGGL((AdapSgdFeatureHalf),
                     dim3(grid_size),
                     dim3(block_size),
                     config_->num_features * sizeof(int16_t),
                     gpu_->shader_stream,
                     train_dataset_->GetUserIdDev(),
                     train_dataset_->GetItemIdDev(),
                     train_dataset_->GetRatingHalfDev(),
                     config_->num_features,
                     config_->learning_rate.f16,
                     config_->regularization_factor.f16,
                     user_features_->GetFeatureHalfDev(),
                     item_features_->GetFeatureHalfDev(),
                     user_features_->GetGradientHalfDev(),
                     item_features_->GetGradientHalfDev());
  logger_->Info(stderr, "Kernels launched\n");
  return 0;
}

int MatrixFactorizer::LaunchAdapSgdFeatureKernel() {
  hipError_t e = hipSuccess;
  e = hipStreamSynchronize(gpu_->h2d_stream);
  logger_->CheckHipError(e);
  dim3 grid_size(train_dataset_->GetNumRecords(), 1, 1);
  dim3 block_size(config_->num_features, 1, 1);
  logger_->Info(stderr, "Launching kernels...\n");
  logger_->StartTimer();
  hipLaunchKernelGGL((AdapSgdFeature),
                     dim3(grid_size),
                     dim3(block_size),
                     config_->num_features * sizeof(float),
                     gpu_->shader_stream,
                     train_dataset_->GetUserIdDev(),
                     train_dataset_->GetItemIdDev(),
                     train_dataset_->GetRatingDev(),
                     config_->num_features,
                     config_->learning_rate.f32,
                     config_->regularization_factor.f32,
                     user_features_->GetFeatureDev(),
                     item_features_->GetFeatureDev(),
                     user_features_->GetGradientDev(),
                     item_features_->GetGradientDev());
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
  if (gpu_) {
    user_features_->CopyToCpu();
    item_features_->CopyToCpu();
  }
  user_features_->DumpToFile();
  item_features_->DumpToFile();
  return 0;
}

float MatrixFactorizer::ComputeSquareErrorSum(Dataset* dataset) {
  std::vector<float> square_errors(dataset->GetNumRecords(), 0);
  const int num_threads = 64;
  std::thread* square_error_workers = new std::thread[num_threads];
  for (int thread_id = 0; thread_id < num_threads; ++thread_id) {
    square_error_workers[thread_id] = std::thread(&MatrixFactorizer::ComputeSquareErrors, this, dataset, thread_id, num_threads, std::ref(square_errors));
  }
  for (int thread_id = 0; thread_id < num_threads; ++thread_id) {
    square_error_workers[thread_id].join();
  }
  delete[] square_error_workers;
  // Kahan sum algorithm
  float error = 0;
  float c = 0;
  for (int record_id = 0; record_id < dataset->GetNumRecords(); ++record_id) {
    float y = square_errors[record_id] - c;
    float t = error + y;
    c = t - error - y;
    error = t;
  }
  return error;
}

void MatrixFactorizer::ComputeSquareErrors(const Dataset* dataset, const int thread_id, const int num_threads, std::vector<float>& square_errors) {
  for (int record_id = thread_id; record_id < dataset->GetNumRecords(); record_id += num_threads) {
    const int user_id = dataset->GetUserId(record_id);
    const int item_id = dataset->GetItemId(record_id);
    float e = 0;
    e = PredictRating(user_id, item_id) - dataset->GetRating(record_id);
    e *= e;
    square_errors[record_id] = e;
  }
}

float MatrixFactorizer::PredictRating(const int user_id, const int item_id) {
  if (!gpu_) {
    const auto& user_feature_vector_begin = user_features_->GetFeatureVector().begin() + user_id * config_->num_features;
    const auto& item_feature_vector_begin = item_features_->GetFeatureVector().begin() + item_id * config_->num_features;
    return std::inner_product(user_feature_vector_begin,
                              user_feature_vector_begin + config_->num_features,
                              item_feature_vector_begin,
                              0.0);
  }
  if (config_->precision == ConfigurationSet::kPrecisionHalf) {
    const auto& user_feature_vector_begin = user_features_->GetFeatureHalfVector().begin() + user_id * config_->num_features;
    const auto& item_feature_vector_begin = item_features_->GetFeatureHalfVector().begin() + item_id * config_->num_features;
    return std::inner_product(user_feature_vector_begin,
                              user_feature_vector_begin + config_->num_features,
                              item_feature_vector_begin,
                              0.0,
                              std::plus<double>(),
                              [](int16_t a, int16_t b) -> double {
                                VariedPrecisionFloat v(a);
                                VariedPrecisionFloat w(b);
                                return v.f32 * w.f32;
                              });
  } else if (config_->precision == ConfigurationSet::kPrecisionSingle) {
    const auto& user_feature_vector_begin = user_features_->GetFeatureVector().begin() + user_id * config_->num_features;
    const auto& item_feature_vector_begin = item_features_->GetFeatureVector().begin() + item_id * config_->num_features;
    return std::inner_product(user_feature_vector_begin,
                              user_feature_vector_begin + config_->num_features,
                              item_feature_vector_begin,
                              0.0);
  }
  return 0;
}

