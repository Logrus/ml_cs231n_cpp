#ifndef SIMPLENEURALNET_H
#define SIMPLENEURALNET_H
#include <vector>
#include <random>
#include <chrono>
#include <algorithm>
#include <assert.h>
#include "CMatrix.h"

typedef std::vector< std::vector<float> > vvfloat;
typedef std::vector<int> vint;

class SimpleNeuralNet
{
public:
    SimpleNeuralNet(int input_size, int hidden_size, int output_size, float std);

    //
    float learning_rate;
    float lambda;

    // Variables for clarity
    int input_size_;
    int hidden_size_;
    int output_size_;
    float std_;

    float loss(const vvfloat &images, const vint &labels, const vint &batch_idx);
    float loss_one_image(const std::vector<float> &image, const int &y);

    int inference(const std::vector<float> &image);

    void initializeW();

    // Weights
    CMatrix<float> W1, b1;
    CMatrix<float> W2, b2;

    // Gradients
    CMatrix<float> dW1, db1;
    CMatrix<float> dW2, db2;
};

#endif // SIMPLENEURALNET_H
