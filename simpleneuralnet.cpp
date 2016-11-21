#include "simpleneuralnet.h"

SimpleNeuralNet::SimpleNeuralNet(int input_size, int hidden_size, int output_size, float std) :
    input_size_(input_size), hidden_size_(hidden_size), output_size_(output_size), std_(std)
{
    // Initialize hyperparameters
    learning_rate = 1e-4;
    lambda = 0.5;
    mu = 0.99;
    neural_statistics.resize(hidden_size_);
    H_.resize(hidden_size_);
    initializeW();
}

float SimpleNeuralNet::loss(const vvfloat &images, const vint &labels, const vint &batch_idx)
{
    // Reset gradience between batches
    dW1.fill(0.f);
    dW2.fill(0.f);
    std::fill(db1.begin(), db1.end(), 0.f);
    std::fill(db2.begin(), db2.end(), 0.f);

    // ************************
    //  Compute loss for batch
    // ************************
    int N = batch_idx.size(); // Batch size
    float L = 0; // Accumulated loss for the batch
    for(int i=0; i<N; ++i){
        L += loss_one_image(images[batch_idx[i]], labels[batch_idx[i]]);
    }
    L /= N;

    // Regularize loss
    float W1_reg(0), W2_reg(0);
    for(int i=0; i<input_size_; ++i){
        for(int h=0; h<hidden_size_; ++h){
            W1_reg += W1(h, i) * W1(h, i);
        }
    }
    for(int s=0; s<output_size_; ++s){
        for(int h=0; h<hidden_size_; ++h){
            W2_reg += W2(s, h) * W2(s, h);
        }
    }

    L +=  0.5 * lambda * W1_reg +  0.5 * lambda * W2_reg;

    // Add gradient regularization
    for(int s=0; s<output_size_; ++s){
        for(int h=0; h<hidden_size_; ++h){
            dW2(s,h) = dW2(s,h)/static_cast<float>(N) + lambda * W2(s, h);
        }
    }
    for(int i=0; i<input_size_; ++i){
        for(int h=0; h<hidden_size_; ++h){
            dW1(h,i) = dW1(h,i)/static_cast<float>(N) + lambda * W1(h, i);
        }
    }

    for(int i=0; i<db2.size(); ++i){
        db2[i] /= static_cast<float>(N);
    }

    for(int i=0; i<db1.size(); ++i){
        db1[i] /= static_cast<float>(N);
    }

    // **********************
    //  Update weights
    // **********************
    for(int s=0; s<output_size_; ++s){
        for(int h=0; h<hidden_size_; ++h){
            //vW2(s, h) = mu*vW2(s, h) - learning_rate*dW2(s, h);
            //W2(s,h) += vW2(s, h);
            W2(s,h) += -learning_rate*dW2(s, h);
        }
    }
    for(int i=0; i<b2.size(); ++i){
        //vb2[i] = mu*vb2[i] - learning_rate*db2[i];
        //b2[i] += vb2[i];
        b2[i] += - learning_rate*db2[i];
    }
    for(int i=0; i<input_size_; ++i){
        for(int h=0; h<hidden_size_; ++h){
            //vW1(h,i) = mu*vW1(h,i) - learning_rate*dW1(h,i);
            //W1(h,i) += vW1(h,i);
            W1(h,i) += - learning_rate*dW1(h,i);
        }
    }
    for(int i=0; i<b1.size(); ++i){
        //vb1[i] = mu*vb1[i] - learning_rate*db1[i];
        //b1[i] += vb1[i];
        b1[i] += - learning_rate*db1[i];
    }

    return L;
}


float SimpleNeuralNet::loss_one_image(const std::vector<float> &image, const int &y){

    // **********************
    //  Compute forward pass
    // **********************
    // H = W1*x + b1
    std::vector<float> H (hidden_size_);
    for(int h=0; h < hidden_size_; ++h){
        for(int i=0; i < input_size_; ++i) {
            H[h] += W1(h, i)*image[i] + b1[h];
        }
    }

    // ReLu and it's derivative
    std::vector<float> dmax (hidden_size_, 1.f);
    for(int h=0; h<hidden_size_; ++h){
        // ReLU
        if(H[h] < 0.0f) {
            H[h] = 0.0f;
            dmax[h] = 0.0f;
        }
        // Uncomment for leaky  ReLU
        // if(H[h] < 0.01f*H[h]) {
        //     H[h] = 0.01f*H[h];
        //     dmax[h] = 0.01f;
        // }
    }

    // S = W2*H + b2
    std::fill(S.begin(), S.end(), 0.f);
    for(int s=0; s < output_size_; ++s){
        for(int h=0; h < hidden_size_; ++h) {
            S[s] += W2(s, h)*H[h] + b2[s];
        }
    }

    // ***********************
    //  Compute loss function
    // ***********************

    // Subtract from scores max to make our computations
    // numerically stable
    std::vector<float> nscores(S);
    float max = *std::max_element(nscores.begin(), nscores.end());
    for(auto &a : nscores) a -= max;

    // Our data is unnormalized log probabilities
    // Exponentiate it
    std::vector<float> power(nscores);
    for(auto &a : power){
        a = std::exp(a);
    }

    // Normalize it
    float normalizer = std::accumulate(power.begin(), power.end(), 0.f);
    std::vector<float> prob(power);
    for(auto &a : prob){
        a /= normalizer;
    }

    float loss = -log( prob[y] );

    // ***********************
    //  Compute backward pass
    // ***********************

    // Compute gradient for loss
    prob[y] -= 1;

    // Propagate it into W2
    // dW2 = H*dscores
    for(int s=0; s<output_size_; ++s){
        for(int h=0; h<hidden_size_; ++h){
            dW2(s, h) += H[h] * prob[s];
        }
    }

    // Propagate it into b2
    // db2 = 1*dscores
    for(int s=0; s<output_size_; ++s){
        db2[s] += prob[s];
    }

    // Propagate gradient further through nonlinearity
    // dhidden = W2*dscores if H was non-negative
    std::vector<float> dhidden(hidden_size_, 0.f);
    for(int s=0; s<output_size_; ++s){
        for(int h=0; h<hidden_size_; ++h){
            dhidden[h] += W2(s, h)*prob[s]*dmax[h];
        }
    }


    // Propagate into W1
    for(int i=0; i<input_size_; ++i){
        for(int h=0; h<hidden_size_; ++h){
            dW1(h,i) += dhidden[h]*image[i];
        }
    }

    // Propagate into b1
    for(int h=0; h<hidden_size_; ++h){
        db1[h] += dhidden[h];
    }


    return loss;
}

int SimpleNeuralNet::inference(const std::vector<float> &image)
{
    std::vector<float> scores = inference_scores(image);

    return std::max_element(scores.begin(), scores.end()) - scores.begin();
}

std::vector<float> SimpleNeuralNet::inference_scores(const std::vector<float> &image)
{
    // **********************
    //  Compute forward pass
    // **********************
    // H = W1*x + b1
    std::vector<float> H (hidden_size_);
    for(int h=0; h < hidden_size_; ++h){
        for(int i=0; i < input_size_; ++i) {
            H[h] += W1(h, i)*image[i] + b1[h];
        }
    }

    // ReLU
    for(int h=0; h < hidden_size_; ++h){
        // ReLU
        H[h] = std::max(0.0f, H[h]);
        // Leaky ReLU
        //H[h] = std::max(0.01f*H[h], H[h]);
        neural_statistics[h] += (H[h]>0);
    }
    this->H_ = H;

    // S = W2*H + b2
    std::fill(S.begin(), S.end(), 0.f);
    for(int s=0; s < output_size_; ++s){
        for(int h=0; h < hidden_size_; ++h) {
            S[s] += W2(s, h)*H[h] + b2[s];
        }
    }

    return S;
}

void SimpleNeuralNet::initializeW()
{
    // Create random generator
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);
    std::normal_distribution<float> distribution(0,std_);

    // Initialize weights and biases
    W1.setSize(hidden_size_, input_size_);
    b1.resize(hidden_size_);
    std::fill(b1.begin(), b1.end(), 0.001);

    // Random init of W1
    for(int x=0; x < W1.xSize(); ++x){
        for(int y=0; y < W1.ySize(); ++y)
        {
            W1(x,y) = distribution(generator);
        }
    }

    W2.setSize(output_size_, hidden_size_);
    b2.resize(output_size_);
    std::fill(b2.begin(), b2.end(), 0.001);

    // Random init of W2
    for(int x=0; x < W2.xSize(); ++x){
        for(int y=0; y < W2.ySize(); ++y)
        {
            W2(x,y) = distribution(generator);
        }
    }

    // Initialize gradients
    dW1.setSize(hidden_size_, input_size_);
    dW2.setSize(output_size_, hidden_size_);
    dW1.fill(0.f);
    dW2.fill(0.f);

    db1.resize(hidden_size_);
    db2.resize(output_size_);
    std::fill(db1.begin(), db1.end(), 0.f);
    std::fill(db2.begin(), db2.end(), 0.f);

    // Initialize score
    S.resize(output_size_);

    // Initialize momentum
    vW1.setSize(hidden_size_, input_size_);
    vW2.setSize(output_size_, hidden_size_);
    vW1.fill(0.f);
    vW2.fill(0.f);
    vb1.resize(hidden_size_);
    vb2.resize(output_size_);
    std::fill(vb1.begin(), vb1.end(), 0.f);
    std::fill(vb2.begin(), vb2.end(), 0.f);
}
