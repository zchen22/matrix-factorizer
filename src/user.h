#ifndef USER_H_
#define USER_H_

// C++ headers
#include <map>
#include <string>

class User {
 public:
  User(const int id);
  ~User();
  // Getters
  int GetId() const { return id_; }
  const std::map<int, float>& GetItemRatingMap() const {
    return item_rating_map_;
  }
  // Add a (item-id, rating) pair
  int AddItemRatingPair(const int item_id, const float rating);
  // Print to string
  std::string ToString();
 private:
  int id_;
  std::map<int, float> item_rating_map_;
};

#endif

