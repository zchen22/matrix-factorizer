#include "minifp.h"

// C++ headers
#include <cassert>

MiniFp::MiniFp(const float f) {
  union {
    float f32;
    uint32_t u32;
  } value;
  value.f32 = f;
  bits_.sign = (value.u32 >> 31) & 1;
  int biased_exponent = (value.u32 >> 23) & 0xff;
  if (biased_exponent == 0) { // +-0 and denormal numbers
    bits_.exponent = 0;
    bits_.mantissa = (value.u32 >> 20) & 0x7;
  } else if (biased_exponent == 0xff) { // +-infinity and NaN
    bits_.exponent = 0xf;
    bits_.mantissa = 0x4; // Quiet NaN
  } else { // Normalized values
    int real_exponent = biased_exponent - 127;
    assert(real_exponent + 7 > 0 && real_exponent + 7 < 0xf);
    bits_.exponent = real_exponent + 7;
    bits_.mantissa = (value.u32 >> 20) & 0x7;
  }
}

MiniFp::MiniFp(const double d) {
  union {
    double f64;
    uint64_t u64;
  } value;
  value.f64 = d;
}

MiniFp::~MiniFp() {
}


