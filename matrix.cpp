#include "matrix.h"

Matrix::Matrix(int rows, int cols) : ySize(rows), xSize(cols)
{
    data.resize(rows*cols);
}

float& Matrix::operator()(const int idx){
    return data[idx];
}

float const & Matrix::operator()(const int idx) const {
    return data[idx];
}

float& Matrix::operator()(const int row, const int col){
    return data[col + row*xSize];
}

float const & Matrix::operator()(const int row, const int col) const {
    return data[col + row*xSize];
}

int Matrix::size(){
    return xSize*ySize;
}

Matrix Matrix::normalize(float lower, float higher){
    Matrix res(xSize, ySize);

    float maxval = std::numeric_limits<float>::min();
    float minval = std::numeric_limits<float>::max();

    // Find max/min
    for(int i=0; i<xSize*ySize; ++i){
        if(data[i] > maxval){
            maxval = data[i];
        }
        if(data[i] < minval){
            minval = data[i];
        }
    }

    // Normalize
    float range = maxval-minval;
    for(int i=0; i<xSize*ySize; ++i){
        res.data[i] = ((data[i]-minval)/range)*higher;
    }

    return res;
}
