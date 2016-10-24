#include "user.h"

// C++ headers
#include <sstream>

User::User(const int id) : id_(id) {
}

User::~User() {
}

int User::AddItemRatingPair(const int item_id, const float rating) {
  item_rating_map_[item_id] = rating;
  return 0;
}

std::string User::ToString() {
  std::stringstream out;
  out << id_ << " " << item_rating_map_.size() << " ";
  for (const auto& item_rating_pair : item_rating_map_) {
    out << item_rating_pair.first << " " << item_rating_pair.second;
  }
  return out.str();
}

