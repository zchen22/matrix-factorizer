#include "record.h"

Record::Record() : user_id(-1), item_id(-1), rating(VariedPrecisionFloat()) {
}

Record::Record(const int user_id, const int item_id, const float rating)
    : user_id(user_id),
      item_id(item_id),
      rating(VariedPrecisionFloat(rating)) {
}

Record::~Record() {
}

