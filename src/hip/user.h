#ifndef USER_H_
#define USER_H_

// C++ headers
#include <string>
#include <unordered_map>

struct User {
 public:
  User(const int id);
  ~User();
  std::string ToString();
 public:
  int id;
  std::unordered_map<int, float> item_rating_map;
};

#endif

