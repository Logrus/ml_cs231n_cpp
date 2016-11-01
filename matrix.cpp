#include "matrix.h"

Matrix::Matrix(int col, int row) : ySize(row), xSize(col)
{
    data.resize(row*col);
}

float& Matrix::operator()(const int col, const int row){
    return data[col + row*xSize];
}

int Matrix::size(){
    return xSize*ySize;
}
