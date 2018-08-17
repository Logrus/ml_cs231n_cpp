#include <classifiers/fisheryatesshuffle.h>
#include <gtest/gtest.h>

#include <iostream>
#include <set>
#include <vector>

TEST(FisherYatesShuffle, GettingRandom500Indexies) {
  FisherYatesShuffle shuffler(50000);

  std::set<size_t> test_set;
  for (int i = 0; i < 5; i++) {
    const std::vector<size_t> res = shuffler.getRandomIndexies(500);
    EXPECT_EQ(res.size(), 500);
    for (const size_t a : res) {
      EXPECT_EQ(test_set.find(a), test_set.end());
      test_set.insert(a);
    }
  }
}
