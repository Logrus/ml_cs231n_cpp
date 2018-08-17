#include <classifiers/fisheryatesshuffle.h>
#include <cassert>

FisherYatesShuffle::FisherYatesShuffle(const size_t n_elements)
    : n_elements_(n_elements), gen_(r_()) {
  setNelem(n_elements_);
}

void FisherYatesShuffle::setNelem(const size_t n_elements) {
  n_elements_ = n_elements;
  // Initialize internal vector from 0 to n_elements-1
  indexies_.resize(n_elements_);
  std::iota(indexies_.begin(), indexies_.end(), 0);
}

std::vector<size_t> FisherYatesShuffle::getRandomIndexies(const int n) {
  std::vector<size_t> result;
  for (int i = n; i != 0; --i) {
    // If we reached the end of unique elements, reinit alg
    if (max_ == 0) max_ = n_elements_;

    // Get a random number between 0 and max_-1
    std::uniform_int_distribution<size_t> uniform_dist(0, max_ - 1);

    const size_t index = static_cast<size_t>(uniform_dist(gen_));
    assert(index >= 0 && index < indexies_.size());

    result.emplace_back(indexies_[index]);

    // Remove this index from next round
    swap(indexies_[index], indexies_[max_ - 1]);
    max_--;
  }
  return result;
}

void FisherYatesShuffle::swap(size_t& a, size_t& b) {
  const size_t c = b;
  b = a;
  a = c;
}
