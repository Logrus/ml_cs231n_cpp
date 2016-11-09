#ifndef FISHERYATESSHUFFLE_H
#define FISHERYATESSHUFFLE_H
#include <algorithm>
#include <vector>
#include <random>
#include <iostream>

/**
 * @brief The FisherYatesShuffle class
 * More info about it can be found here:
 * https://en.wikipedia.org/wiki/Fisher%E2%80%93Yates_shuffle
 * And here:
 * http://stackoverflow.com/a/196065
 */

class FisherYatesShuffle
{
public:
    FisherYatesShuffle() {} // Default constructor
    FisherYatesShuffle(int n_elements);

    void set_nelem(int n_elements);

    std::vector<int> get_random_indexies(int n);

    void swap(int &a, int &b);

private:
    std::vector<int> indexies_;
    int max_, n_elements_;
    std::random_device r_;
};

#endif // FISHERYATESSHUFFLE_H
