#ifndef MATRIX_H
#define MATRIX_H
#include <vector>

class Matrix
{
public:
    Matrix(int, int);

    float& operator()(const int col, const int row);

    std::vector<float> data;
    int xSize;
    int ySize;

    int size();
};

#endif // MATRIX_H
