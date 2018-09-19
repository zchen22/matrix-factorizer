#ifndef VARIED_PRECISION_FLOAT_H_
#define VARIED_PRECISION_FLOAT_H_

// C++ headers
#include <cstdint>

struct VariedPrecisionFloat {
 public:
  VariedPrecisionFloat();
  VariedPrecisionFloat(const int16_t h);
  VariedPrecisionFloat(const float f);
  ~VariedPrecisionFloat();
  int Half2Float(const int16_t h);
  int Float2Half(const float f);
 public:
  union {
    uint16_t u16;
    int16_t f16;
  };
  union {
    uint32_t u32;
    float f32;
  };
};

#endif

