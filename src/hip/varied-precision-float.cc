#include "varied-precision-float.h"

// C++ headers
#include <cassert>
#include <cstdio>

VariedPrecisionFloat::VariedPrecisionFloat() {
  u16 = 0;
  u32 = 0;
}

VariedPrecisionFloat::VariedPrecisionFloat(const int16_t h)
    : f16(h) {
  Half2Float(h);
}

VariedPrecisionFloat::VariedPrecisionFloat(const float f)
    : f32(f) {
  Float2Half(f);
}

VariedPrecisionFloat::~VariedPrecisionFloat() {
}

int VariedPrecisionFloat::Half2Float(const int16_t h) {
  const uint32_t sign = h >> 15;
  const uint32_t exp = (h >> 10) & 0x1f;
  const uint32_t frac = h & 0x3ff;
  u32 = 0;
  if (exp == 0) {
    u32 = (sign << 31) | (frac << 13);
  } else if (exp < 0x1f) {
    u32 = (sign << 31) | ((exp + 112) << 23) | (frac << 13);
  } else {
    u32 = (sign << 31) | (0xff << 23) | (frac << 13);
  }
  return 0;
}

int VariedPrecisionFloat::Float2Half(const float f) {
  const uint16_t sign = u32 >> 31;
  const int16_t exp = ((u32 >> 23) & 0xff);
  const uint16_t frac = u32 & 0x3fffff;
  f16 = 0;
  if (exp == 0) {
    f16 = (sign << 15) | (frac >> 13);
  } else if (exp < 0xff) {
    int16_t real_exp = exp - 127;
    if (real_exp < -14) {
      real_exp = -14;
    } else if (real_exp > 15) {
      real_exp = 15;
    }
    f16 = (sign << 15) | ((real_exp + 15) << 10) | (frac >> 13);
  } else {
    f16 = (sign << 15) | (0x1f << 10) | ((frac? 0x200000 : 0) >> 13);
  }
  return 0;
}

