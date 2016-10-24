#ifndef GPU_H_
#define GPU_H_

// CUDA headers
#include <cuda_runtime.h>

// Project headers
#include "logger.h"

class Gpu {
 public:
  Gpu(const int id, Logger* logger);
  ~Gpu();
  // Getters
  cudaStream_t GetShaderStream() const { return shader_stream_; }
  cudaStream_t GetH2dStream() const { return h2d_stream_; }
  cudaStream_t GetD2hStream() const { return d2h_stream_; }
 private:
  // ID
  int id_;
  // Streams
  cudaStream_t shader_stream_;
  cudaStream_t h2d_stream_;
  cudaStream_t d2h_stream_;
  // Logger
  Logger* logger_;
};

#endif

