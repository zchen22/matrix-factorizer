#ifndef COMMON_H_
#define COMMON_H_

union Reg32 {
  unsigned int u32;
  float f32;
};
unsigned short Float2Half(const float f);
float Half2Float(const unsigned short h);

#endif

