#ifndef CONFIGURATION_SET_H_
#define CONFIGURATION_SET_H_

// C++ headers
#include <string>

// Project headers
#include "logger.h"

struct ConfigurationSet {
 public:
  ConfigurationSet(Logger* logger);
  ~ConfigurationSet();
  int Load(const std::string& filename);
  std::string ToString() const;
 public:
  // Learning parameters
  enum GdMode {
    kGdModeGd = 0,
    kGdModeMiniBatchSgd,
    kGdModeAdapSgd
  } gd_mode;
  int batch_size;
  int num_features;
  float learning_rate;
  float regularization_factor;
  int max_num_iterations;
  enum DecompMode {
    kDecompModeRecord = 0,
    kDecompModeRow,
    kDecompModeFeature
  } decomp_mode;
  bool lock;
  enum Precision {
    kPrecisionMini = 0,
    kPrecisionHalf,
    kPrecisionSingle,
    kPrecisionDouble
  } precision;
  bool show_train_rmse;
  bool show_test_rmse;
  // Logger
  Logger* logger;
};

#endif

