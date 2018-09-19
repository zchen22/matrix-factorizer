#include "gpu.h"

// C++ headers
#include <cassert>

Gpu::Gpu(const int id, Logger* logger) : id(id), logger(logger) {
  hipError_t e = hipSuccess;
  int num_gpus = 0;
  e = hipGetDeviceCount(&num_gpus);
  logger->CheckHipError(e);
  assert(num_gpus > 0);
  e = hipSetDevice(id);
  logger->CheckHipError(e);
  e = hipStreamCreate(&shader_stream);
  logger->CheckHipError(e);
  e = hipStreamCreate(&h2d_stream);
  logger->CheckHipError(e);
  e = hipStreamCreate(&d2h_stream);
  logger->CheckHipError(e);
}

Gpu::~Gpu() {
}

