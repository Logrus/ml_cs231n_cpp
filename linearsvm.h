#ifndef LINEARSVM_H
#define LINEARSVM_H
#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <algorithm>
#include <assert.h>
#include "CMatrix.h"

class LinearSVM
{
public:
    LinearSVM(int, int);

    CMatrix<float> W;
    CMatrix<float> dW;

    int C; // classes
    int D; // data dimentionality

    float lambda;
    float learning_rate;

    /**
     * @brief Computes SVM L2 regularization term W^2
     * @return float representing elementwise sum of squares of weight matrix elements
     */
    float L2W_reg();

    float loss_one_image(const std::vector<int> &image, const int &y);
    float loss(const std::vector< std::vector<int> > &images, const std::vector<int> &labels, int from, int to);

    int inference(const std::vector<int> &image);

};

#endif // LINEARSVM_H
