#ifndef CLASSIFIER_H
#define CLASSIFIER_H
#include "CMatrix.h"

/**
 * @brief The Abstract Base Classifier class
 */
class Classifier
{
public:

    Classifier(int classes, int dimentionality);

    virtual void initializeW() = 0;
    virtual float L2W_reg() = 0;
    virtual float loss_one_image(const std::vector<float> &image, const int &y) = 0;
    virtual float loss(const std::vector< std::vector<float> > &images, const std::vector<int> &labels, int from, int to) = 0;
    virtual int inference(const std::vector<float> &image) = 0;

    virtual void copyW(const CMatrix<float> W);

    virtual std::vector<float> scores(const std::vector<float> &image);

    CMatrix<float> W;
    CMatrix<float> dW;

    int C; // classes
    int D; // data dimentionality

    float lambda;
    float learning_rate;

    virtual ~Classifier() {};
};

#endif // CLASSIFIER_H
