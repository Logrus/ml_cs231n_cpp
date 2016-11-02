#ifndef MATRIX_H
#define MATRIX_H
#include <vector>
#include <limits>

class Matrix
{
public:
    Matrix(int, int);

    float& operator()(const int idx);
    float const & operator()(const int idx) const;
    float& operator()(const int col, const int row);
    float const & operator()(const int col, const int row) const;

    std::vector<float> data;
    int xSize;
    int ySize;

    int size();

    Matrix normalize(int lower, int higher);
};

#endif // MATRIX_H
