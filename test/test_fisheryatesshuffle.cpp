#include <iostream>
#include <unordered_set>
#include <vector>
#undef NDEBUG  // Do assert always
#include <assert.h>
#include <classifiers/fisheryatesshuffle.h>

int main() {
  FisherYatesShuffle shuffler(50000);

  std::vector<size_t> res;
  std::unordered_set<size_t> test_set;

  for (int i = 0; i < 5; i++) {
    res = shuffler.getRandomIndexies(500);

    for (auto a : res) {
      assert(test_set.find(a) == test_set.end());
      test_set.insert(a);
    }
  }
  std::cout << "Test passed! " << std::endl;
  return 0;
}
