#ifndef MINIFP_H_
#define MINIFP_H_

// C++ headers
#include <cstdint>

class MiniFp {
 public:
  MiniFp(const float f);
  MiniFp(const double d);
  ~MiniFp();
  // Getters
  int GetSign() const { return bits_.sign; }
  int GetExponent() const { return bits_.exponent; }
  int GetMantissa() const { return bits_.mantissa; }
  int GetValueAsUint() const { return value_.as_uint; }
  int GetValueAsFloat() const { return value_.as_float; }
 private:
  struct {
    int8_t sign : 1;
    int8_t exponent : 4;
    int8_t mantissa : 3;
  } bits_;
  union {
    uint8_t as_uint;
    float as_float;
  } value_;
};

#endif

