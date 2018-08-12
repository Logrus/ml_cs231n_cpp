#ifndef LINEARSOFTMAX_H
#define LINEARSOFTMAX_H
#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <assert.h>
#include <uni_freiburg_cv/CMatrix.h>
#include "classifier.h"
#include <numeric>

class LinearSoftmax : public Classifier
{
public:
    LinearSoftmax(int,int);

    /**
     * @brief Computes SVM L2 regularization term W^2
     * @return float representing elementwise sum of squares of weight matrix elements
     */
    float L2W_reg();

    float loss_one_image(const std::vector<float> &image, const int &y);
    float loss(const std::vector< std::vector<float> > &images, const std::vector<int> &labels, const std::vector<int> &indexies);

    int inference(const std::vector<float> &image);

    std::vector<float> inference_loss(const std::vector<float> &image, const int &y);

};

#endif // LINEARSOFTMAX_H
