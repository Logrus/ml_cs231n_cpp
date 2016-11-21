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
    float mu; // momentum

    // Variables for clarity
    int input_size_;
    int hidden_size_;
    int output_size_;
    float std_;

    float loss(const vvfloat &images, const vint &labels, const vint &batch_idx);
    float loss_one_image(const std::vector<float> &image, const int &y);

    int inference(const std::vector<float> &image);

    std::vector<float> inference_scores(const std::vector<float> &image);

    void initializeW();
    bool saveWeights(std::string filename);
    bool loadWeights(std::string filename);

    // Weights
    CMatrix<float> W1;
    CMatrix<float> W2;

    std::vector<float> b1, b2;

    // Gradients
    CMatrix<float> dW1;
    CMatrix<float> dW2;

    std::vector<float> db1, db2;

    // Velocities for momentum
    CMatrix<float> vW1;
    CMatrix<float> vW2;
    std::vector<float> vb1, vb2;

    // For Adam
    CMatrix<float> mW1;
    CMatrix<float> mW2;
    std::vector<float> mb1, mb2;
    int t;
    float beta1, beta2;

    // Scores
    std::vector<float> S;

    // Statistics
    std::vector<float> H_;
    std::vector<float> neural_statistics;
};

#endif // SIMPLENEURALNET_H
