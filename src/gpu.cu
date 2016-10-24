#include "gpu.h"

Gpu::Gpu(const int id, Logger* logger) : id_(id), logger_(logger) {
  cudaError_t e = cudaSuccess;
  e = cudaStreamCreate(&shader_stream_);
  logger_->CheckCudaError(e);
  e = cudaStreamCreate(&h2d_stream_);
  logger_->CheckCudaError(e);
  e = cudaStreamCreate(&d2h_stream_);
  logger_->CheckCudaError(e);
}

Gpu::~Gpu() {
}

