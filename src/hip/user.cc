#include "user.h"

// C++ headers
#include <sstream>

User::User(const int id) : id(id) {
}

User::~User() {
}

std::string User::ToString() {
  std::stringstream out;
  out << id << " " << item_rating_map.size() << " ";
  for (const auto& item_rating_pair : item_rating_map) {
    out << item_rating_pair.first << " " << item_rating_pair.second;
  }
  return out.str();
}

