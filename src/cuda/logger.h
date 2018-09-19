#ifndef LOGGER_H_
#define LOGGER_H_

// C++ headers
#include <cstdarg>
#include <string>

// Linux headers
#include <sys/time.h>

// CUDA headers
#include <cuda_runtime.h>

class Logger {
 public:
  Logger();
  ~Logger();
  // Getters
  double ReadTimer() const {
    return (1. * (end_time_.tv_sec * 1e6 + end_time_.tv_usec) -
      (start_time_.tv_sec * 1e6 + start_time_.tv_usec)) / 1e6;
  }
  // Logging
  int Warning(const char* format, ...);
  int Info(const char* format, ...);
  int Debug(const char* format, ...);
  int Error(FILE* file, const char* format, ...);
  int Warning(FILE* file, const char* format, ...);
  int Info(FILE* file, const char* format, ...);
  int Debug(FILE* file, const char* format, ...);
  // Error checking
  int CheckCudaError(const cudaError_t e);
  // Timing
  int StartTimer();
  int StopTimer();
 private:
  // Log file
  std::string filename_;
  FILE* file_;
  // Timers
  timeval start_time_;
  timeval end_time_;
};

#endif

