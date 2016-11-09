#include "simpleneuralnet.h"

SimpleNeuralNet::SimpleNeuralNet(int input_size, int hidden_size, int output_size, float std) :
    input_size_(input_size), hidden_size_(hidden_size), output_size_(output_size), std_(std)
{
    // Initialize hyperparameters
    learning_rate = 1.e-10;
    lambda = 5.e-3;
}

float SimpleNeuralNet::loss(const vvfloat &images, const vint &labels, const vint &batch_idx)
{
    // Reset gradience between batches
    dW1.fill(0.f);
    db1.fill(0.f);
    dW2.fill(0.f);
    db2.fill(0.f);

    // ************************
    //  Compute loss for batch
    // ************************
    int N = batch_idx.size(); // Batch size
    std::cout << "Batch size " << N << std::endl; std::cin.get();
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
    std::cout << "Loss for one batch " << L << std::endl; std::cin.get();

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
        db2(i,1) /= static_cast<float>(N);
    }

    for(int i=0; i<db1.size(); ++i){
        db1(i,1) /= static_cast<float>(N);
    }

    // **********************
    //  Update weights
    // **********************
//    for(int s=0; s<output_size_; ++s){
//        for(int h=0; h<hidden_size_; ++h){
//            W2(s,h) -= learning_rate * dW2(s, h);
//        }
//    }

//    for(int i=0; i<b2.size(); ++i){
//        b2(i,1) -= learning_rate*db2(i,1);
//    }
//    for(int i=0; i<input_size_; ++i){
//        for(int h=0; h<hidden_size_; ++h){
//            W1(h,i) -= learning_rate * dW1(h, i);
//        }
//    }

//    for(int i=0; i<b1.size(); ++i){
//        b1(i,1) -= learning_rate*db1(i,1);
//    }

    return L;
}


float SimpleNeuralNet::loss_one_image(const std::vector<float> &image, const int &y){

    assert(image.size() == 3072);

    // **********************
    //  Compute forward pass
    // **********************
    // H = W1*x + b1
    std::vector<float> H (hidden_size_, 0);

    for(int h=0; h < hidden_size_; ++h){
        for(int i=0; i < input_size_; ++i) {
            H[h] += W1(h, i) * image[i] + b1(h,1);
        }
        //std::cout << H[h] << " ";  std::cin.get();
    }
    //std::cout << std::endl;

    std::cout << "H: " << std::endl;
    for(int i=0; i<H.size(); ++i){
        std::cout << H[i] << " ";
    }
    std::cin.get();

    // ReLu and it's derivative
    std::vector<float> dmax (hidden_size_, 1.f);
    for(int h=0; h<hidden_size_; ++h){
        if(H[h] <= 0.f) {
            H[h] = 0.f;
            dmax[h] = 0.f;
        }
    }

//    std::cout << "H: " << std::endl;
//    for(int i=0; i<H.size(); ++i){
//        std::cout << H[i] << " ";
//    }
//    std::cin.get();


    // S = W2*H + b2
    std::vector<float> S(output_size_, 0);
    for(int s=0; s < output_size_; ++s){
        for(int h=0; h < hidden_size_; ++h) {
            S[s] += W2(s, h) * H[h] + b2(s,1);
        }
    }

//    std::cout << "W2: " << std::endl;
//    for(int i=0; i<W2.xSize(); ++i){
//        for(int j=0; j<W2.ySize(); ++j){
//            std::cout << W2(i,j) << " ";
//        }
//    }
//    std::cin.get();

//    std::cout << "b2: " << std::endl;
//    for(int i=0; i< b2.xSize(); ++i){
//        for(int j=0; j<b2.ySize(); ++j){
//            std::cout << b2(i,j) << " ";
//        }
//    }
//    std::cin.get();

//    std::cout << "S: " << std::endl;
//    for(int i=0; i<S.size(); ++i){
//        std::cout << S[i] << " ";
//    }
//    std::cin.get();

    // ***********************
    //  Compute loss function
    // ***********************

    // Subtract from scores max to make our computations
    // numerically stable
    float max = *std::max_element(S.begin(), S.end());
    for(auto &a : S) a -= max;

    // Our data is unnormalized log probabilities
    // Exponentiate it
    std::vector<float> power(S);
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
    std::cout << "Loss for one image " << loss << std::endl;
    assert( loss < 3.f );

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
        db2(s, 1) += prob[s];
    }

    // Propagate gradient further through nonlinearity
    // dhidden = W2*dscores if H was non-negative
    std::vector<float> dhidden(hidden_size_, 0.f);
    for(int s=0; s<output_size_; ++s){
        for(int h=0; h<hidden_size_; ++h){
            dhidden[h] += W2(s, h)*prob[s]*dmax[h];
        }
    }
//    std::cout << "dhidden: " << std::endl;
//    for(int i=0; i<dhidden.size(); ++i){
//        std::cout << dhidden[i] << " ";
//    }
//    std::cin.get();


    // Propagate into W1
    for(int i=0; i<input_size_; ++i){
        for(int h=0; h<hidden_size_; ++h){
            dW1(h,i) += dhidden[h]*image[i];
        }
    }

    // Propagate into b1
    for(int h=0; h<hidden_size_; ++h){
        db1(h, 1) += dhidden[h];
    }


    return loss;
}

int SimpleNeuralNet::inference(const std::vector<float> &image)
{
    // **********************
    //  Compute forward pass
    // **********************
    // H = W1*x + b1
    std::vector<float> H (hidden_size_);
    std::vector<float> dmax (hidden_size_, 1);
    for(int h=0; h < hidden_size_; ++h){
        for(int i=0; i < input_size_; ++i) {
            H[h] += std::max(0.f, W1(h, i) * image[i] + b1(h,1) );
        }
    }

    // S = W2*H + b2
    std::vector<float> S(output_size_);
    for(int s=0; s < output_size_; ++s){
        for(int h=0; h < hidden_size_; ++h) {
            S[s] += W2(s, h) * H[h] + b2(s,1);
        }
    }

    return std::max_element(S.begin(), S.end()) - S.begin();
}

void SimpleNeuralNet::initializeW()
{
    // Create random generator
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);
    std::normal_distribution<float> distribution(0,std_);

    // Initialize weights and biases
    W1.setSize(hidden_size_, input_size_);
    b1.setSize(hidden_size_, 1);
    b1.fill(0.001);

    // Random init of W1
    for(int x=0; x < W1.xSize(); ++x){
        for(int y=0; y < W1.ySize(); ++y)
        {
            W1(x,y) = distribution(generator);
        }
    }

    W2.setSize(output_size_, hidden_size_);
    b2.setSize(output_size_, 1);
    b2.fill(0.001);

    // Random init of W2
    for(int x=0; x < W2.xSize(); ++x){
        for(int y=0; y < W2.ySize(); ++y)
        {
            W2(x,y) = distribution(generator);
        }
    }

    // Initialize gradients
    dW1.setSize(hidden_size_, input_size_);
    db1.setSize(hidden_size_, 1);
    dW2.setSize(output_size_, hidden_size_);
    db2.setSize(output_size_ ,1);
    dW1.fill(0.f);
    db1.fill(0.f);
    dW2.fill(0.f);
    db2.fill(0.f);
}
