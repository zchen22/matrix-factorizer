#include "logger.h"

// C++ headers
#include <ctime>

Logger::Logger() {
  // Generate filename
  time_t timer = time(NULL);
  tm local_time = *localtime(&timer);
  char time_string[1024] = {0};
  snprintf(time_string, sizeof time_string,
           "%02d-%02d-%02d-%02d-%02d.log", local_time.tm_mon,
           local_time.tm_mday, local_time.tm_hour, local_time.tm_min,
           local_time.tm_sec);
  filename_ = std::string("mf-") + time_string;
  // Open file
  file_ = fopen(filename_.c_str(), "w");
  if (!file_) {
    Error(stderr, "File '%s' cannot be opened!\n", filename_.c_str());
  }
}

Logger::~Logger() {
  // Close file
  fclose(file_);
}

int Logger::Warning(const char* format, ...) {
  fprintf(file_, "%-8s: ", "WARNING");
  va_list args;
  va_start(args, format);
  vfprintf(file_, format, args);
  va_end(args);
  fflush(file_);
  return 0;
}

int Logger::Info(const char* format, ...) {
  fprintf(file_, "%-8s: ", "INFO");
  va_list args;
  va_start(args, format);
  vfprintf(file_, format, args);
  va_end(args);
  fflush(file_);
  return 0;
}

int Logger::Debug(const char* format, ...) {
  fprintf(file_, "%-8s: ", "DEBUG");
  va_list args;
  va_start(args, format);
  vfprintf(file_, format, args);
  va_end(args);
  fflush(file_);
  return 0;
}

int Logger::Error(FILE* file, const char* format, ...) {
  fprintf(file, "%-8s: ", "ERROR");
  va_list args;
  va_start(args, format);
  vfprintf(file, format, args);
  va_end(args);
  fflush(file);
  exit(EXIT_FAILURE);
}

int Logger::Warning(FILE* file, const char* format, ...) {
  fprintf(file, "%-8s: ", "WARNING");
  va_list args;
  va_start(args, format);
  vfprintf(file, format, args);
  va_end(args);
  fflush(file);
  return 0;
}

int Logger::Info(FILE* file, const char* format, ...) {
  fprintf(file, "%-8s: ", "INFO");
  va_list args;
  va_start(args, format);
  vfprintf(file, format, args);
  va_end(args);
  fflush(file);
  return 0;
}

int Logger::Debug(FILE* file, const char* format, ...) {
  fprintf(file, "%-8s: ", "DEBUG");
  va_list args;
  va_start(args, format);
  vfprintf(file, format, args);
  va_end(args);
  fflush(file);
  return 0;
}

int Logger::CheckCudaError(const cudaError_t e) {
  if (e != cudaSuccess) {
    Error(stderr, "%s\n", cudaGetErrorString(e));
  }
  return 0;
}

int Logger::StartTimer() {
  gettimeofday(&start_time_, NULL);
  return 0;
}

int Logger::StopTimer() {
  gettimeofday(&end_time_, NULL);
  return 0;
}

