#ifndef LOGGER_H_
#define LOGGER_H_

// C++ headers
#include <cstdarg>
#include <string>

// Linux headers
#include <sys/time.h>

// HIP headers
#include <hip/hip_runtime.h>

class Logger {
 public:
  Logger();
  ~Logger();
  // Logging
  int Warning(const char* format, ...);
  int Info(const char* format, ...);
  int Debug(const char* format, ...);
  int Error(FILE* file, const char* format, ...);
  int Warning(FILE* file, const char* format, ...);
  int Info(FILE* file, const char* format, ...);
  int Debug(FILE* file, const char* format, ...);
  // Error checking
  int CheckHipError(const hipError_t e);
  // Timing
  int StartTimer();
  int StopTimer();
  double ReadTimer() const;
 private:
  // Log file
  std::string filename_;
  FILE* file_;
  // Timers
  timeval start_time_;
  timeval end_time_;
};

#endif

