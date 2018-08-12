#include <classifiers/fisheryatesshuffle.h>

FisherYatesShuffle::FisherYatesShuffle(const size_t n_elements) : n_elements_(n_elements) {
  set_nelem(n_elements_);
}

void FisherYatesShuffle::set_nelem(const size_t n_elements) {
  n_elements_ = n_elements;
  // Initialize internal vector from 0 to n_elements-1
  indexies_.resize(n_elements_);
  std::iota(indexies_.begin(), indexies_.end(), 0);
}

std::vector<size_t> FisherYatesShuffle::get_random_indexies(int n) {
  std::vector<size_t> result;
  for (int i = n; i != 0; --i) {
    // If we reached the end of unique elements, reinit alg
    if (max_ == 0) max_ = n_elements_;

    // Get a random number between 0 and max_-1
    std::default_random_engine rand_engine(r_());
    std::uniform_int_distribution<size_t> uniform_dist(0, max_ - 1);

    size_t index = uniform_dist(rand_engine);
    result.push_back(indexies_[index]);

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
