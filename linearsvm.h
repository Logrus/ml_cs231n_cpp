#ifndef LINEARSVM_H
#define LINEARSVM_H
#include <iostream>
#include <vector>
#include <random>
#include "matrix.h"

class LinearSVM
{
public:
    LinearSVM(int, int);

    Matrix W;
    Matrix dW;
    Matrix scores;

    int C; // classes
    int D; // data dimentionality

    float lambda;
    float learning_rate;
    int count;


    /**
     * @brief Computes SVM L2 regularization term W^2
     * @return float representing elementwise square of weight matrix
     */
    float L2W_reg();

    float loss_one_image(const std::vector<int> &image, const int &y);
    float loss(const std::vector< std::vector<int> > &images, const std::vector<int> &labels, int from, int to);
    void updateWeights();

};

#endif // LINEARSVM_H
