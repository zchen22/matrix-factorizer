#include "common.h"

// C++ headers
#include <cassert>

unsigned short Float2Half(const float f) {
  const Reg32 u = { .f32 = f };
  const unsigned int sign = u.u32 >> 31;
  const int exp = ((u.u32 >> 23) & 0xff);
  const unsigned int frac = u.u32 & 0x3fffff;
  unsigned short h = 0;
  if (exp == 0) {
    h = (sign << 15) | (frac >> 13);
  } else if (exp < 0xff) {
    int real_exp = exp - 127;
    if (real_exp < -14) {
      real_exp = -14;
    } else if (real_exp > 15) {
      real_exp = 15;
    }
    h = (sign << 15) | ((real_exp + 15) << 10) | (frac >> 13);
  } else {
    assert(false);
  }
  return h;
}

float Half2Float(const unsigned short h) {
  const unsigned int sign = h >> 15;
  const unsigned int exp = ((h >> 10) & 0x1f);
  const unsigned int frac = h & 0x3ff;
  Reg32 f = { .u32 = 0 };
  if (exp == 0) {
    f.u32 = (sign << 31) | (frac << 13);
  } else if (exp < 0x1f) {
    f.u32 = (sign << 31) | ((exp + 112) << 23) | (frac << 13);
  } else {
    assert(false);
  }
  return f.f32;
}

