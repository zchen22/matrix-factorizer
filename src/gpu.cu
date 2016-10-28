#include "gpu.h"

// C++ headers
#include <cassert>

Gpu::Gpu(const int id, Logger* logger) : id_(id), logger_(logger) {
  cudaError_t e = cudaSuccess;
  int num_gpus = 0;
  e = cudaGetDeviceCount(&num_gpus);
  logger_->CheckCudaError(e);
  assert(num_gpus > 0);
  e = cudaSetDevice(id_);
  logger_->CheckCudaError(e);
  e = cudaStreamCreate(&shader_stream_);
  logger_->CheckCudaError(e);
  e = cudaStreamCreate(&h2d_stream_);
  logger_->CheckCudaError(e);
  e = cudaStreamCreate(&d2h_stream_);
  logger_->CheckCudaError(e);
}

Gpu::~Gpu() {
}

