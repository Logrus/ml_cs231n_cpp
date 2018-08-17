#pragma once
#include <algorithm>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

/**
 * @brief The FisherYatesShuffle class
 * More info about it can be found here:
 * https://en.wikipedia.org/wiki/Fisher%E2%80%93Yates_shuffle
 * And here:
 * http://stackoverflow.com/a/196065
 */

class FisherYatesShuffle {
 public:
  FisherYatesShuffle() {}
  FisherYatesShuffle(const size_t n_elements);

  void setNelem(const size_t n_elements);

  std::vector<size_t> getRandomIndexies(const int n);

 private:
  void swap(size_t& a, size_t& b);
  std::vector<size_t> indexies_;
  size_t max_, n_elements_;
  std::random_device r_;
  std::mt19937 gen_;
};
