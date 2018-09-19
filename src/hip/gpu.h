#ifndef GPU_H_
#define GPU_H_

// HIP headers
#include <hip/hip_runtime.h>

// Project headers
#include "logger.h"

struct Gpu {
 public:
  Gpu(const int id, Logger* logger);
  ~Gpu();
 public:
  int id;
  hipStream_t shader_stream;
  hipStream_t h2d_stream;
  hipStream_t d2h_stream;
  Logger* logger;
};

#endif

