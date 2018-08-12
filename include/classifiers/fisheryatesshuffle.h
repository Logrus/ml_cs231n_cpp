#ifndef FISHERYATESSHUFFLE_H
#define FISHERYATESSHUFFLE_H
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
  FisherYatesShuffle() {}  // Default constructor
  FisherYatesShuffle(const size_t n_elements);

  void set_nelem(const size_t n_elements);

  std::vector<size_t> get_random_indexies(int n);

 private:
  void swap(size_t& a, size_t& b);
  std::vector<size_t> indexies_;
  size_t max_, n_elements_;
  std::random_device r_;
};

#endif  // FISHERYATESSHUFFLE_H
