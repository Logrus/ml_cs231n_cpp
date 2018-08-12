#ifndef CLASSIFIER_H
#define CLASSIFIER_H
#include <uni_freiburg_cv/CMatrix.h>
#include <chrono>
#include <random>

/**
 * @brief The Abstract Base Classifier class
 */
class Classifier
{
public:

    Classifier(int classes, int dimentionality);

    // Pure virtuals
    virtual float L2W_reg() = 0;
    virtual float loss_one_image(const std::vector<float> &image, const int &y) = 0;
    virtual float loss(const std::vector< std::vector<float> > &images, const std::vector<int> &labels, const std::vector<int> &indexies) = 0;
    virtual int inference(const std::vector<float> &image) = 0;
    virtual std::vector<float> inference_loss(const std::vector<float> &image, const int &y) = 0;

    // Virtuals
    virtual void initializeW();
    virtual void copyW(const CMatrix<float> W);
    virtual std::vector<float> scores(const std::vector<float> &image);

    float weight_ratio();

    CMatrix<float> W;
    CMatrix<float> dW;

    int C; // classes
    int D; // data dimentionality

    float lambda;
    float learning_rate;

    virtual ~Classifier() {};
};

#endif // CLASSIFIER_H
