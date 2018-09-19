#ifndef RECORD_H_
#define RECORD_H_

#include "varied-precision-float.h"

struct Record {
 public:
  Record();
  Record(const int user_id, const int item_id, const float rating);
  ~Record();
  int user_id;
  int item_id;
  VariedPrecisionFloat rating;
};

#endif

